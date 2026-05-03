"""Compute TAP-Vid metrics from eval_tapvid_flow.py predictions.

Two-step pipeline:
  1. eval_tapvid_flow.py writes per-shard `tapvid_formatted_results_256res_*.json`
     into <out_dir>/<model_slug>_mask_ratio_X/results/.
  2. This script globs those JSONs, aggregates by video, joins to the official
     TapVID-DAVIS ground truth (from the pickle), and prints the official
     TapVID metrics: AD (avg distance), MD (median distance), Pct (avg %% pts
     within thresh), AJ (avg Jaccard), OA (occlusion accuracy), OF1.

Ported from the internal `tapvid_eval_offline.py` script — only the
`process_json` path; the unused `process_json_cfg`, `sparse_pkl_process`,
and `tapvid_eval` variants are dropped. The TapVID metric implementation
(`compute_tapvid_metrics`) is preserved verbatim.

Usage:
    python -m zwm.eval.flow.grade_tapvid_flow \\
        --root_dir viz/eval/flow/tapvid_davis_first/std_2_zoom_4/awwkl_zwm-babyview-170m_model_mask_ratio_0.9 \\
        --pkl_path /path/to/tapvid_davis.pkl \\
        --occ_thresh 0.4
"""
import json
import os
import pickle
from argparse import ArgumentParser
from collections import Counter, defaultdict
from glob import glob

import numpy as np
from tqdm import tqdm


def compute_tapvid_metrics(
    query_points,
    gt_occluded,
    gt_tracks,
    pred_occluded,
    pred_tracks,
    query_mode,
    evaluation_points=None,
    mask_out_query_point_for_ad=False,
):
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.).

    Verbatim from the TAP-Vid paper reference implementation. Inputs are in
    raster coordinates scaled to a 256x256 image.

    Returns a dict with keys:
        occlusion_accuracy, occ_tp/fp/fn, avg_distance, median_distance,
        pts_within_{1,2,4,8,16}, jaccard_{1,2,4,8,16},
        average_pts_within_thresh, average_jaccard.
    """
    assert gt_occluded.shape == pred_occluded.shape

    metrics = {}

    if evaluation_points is None:
        # Don't evaluate the query point itself.
        one_hot_eye = np.eye(gt_tracks.shape[2])
        query_frame = np.round(query_points[..., 0]).astype(np.int32)
        evaluation_points = one_hot_eye[query_frame] == 0

        if query_mode == "first":
            assert gt_occluded.shape[0] == 1, "Expected batch size 1 gt_occluded"
            for i in range(gt_occluded.shape[1]):
                index = np.where(gt_occluded[0, i] == 0)[0][0]
                evaluation_points[0, i, :index] = False
        elif query_mode != "strided":
            raise ValueError("Unknown query mode " + query_mode)

    # Occlusion accuracy
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc
    metrics["occ_tp"] = np.sum(
        np.equal(pred_occluded, gt_occluded) & gt_occluded & evaluation_points, axis=(1, 2)
    )
    metrics["occ_fp"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded)) & pred_occluded & evaluation_points, axis=(1, 2)
    )
    metrics["occ_fn"] = np.sum(
        np.logical_not(np.equal(pred_occluded, gt_occluded)) & np.logical_not(pred_occluded) & evaluation_points,
        axis=(1, 2),
    )

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    L2_error = np.sqrt(np.sum(np.square(pred_tracks - gt_tracks), axis=-1))

    l2_eval_mask = visible
    if mask_out_query_point_for_ad:
        l2_eval_mask = l2_eval_mask & evaluation_points

    masked_L2_error = L2_error * l2_eval_mask
    nonzero_masked_error = L2_error[l2_eval_mask.astype(bool)]
    assert np.allclose(masked_L2_error.sum(), nonzero_masked_error.sum())
    assert nonzero_masked_error.size == np.sum(l2_eval_mask)
    # When all eval points in this video are occluded, l2_eval_mask sums to 0
    # and the source's `np.sum(masked) / np.sum(mask)` divides by zero. Only
    # surfaces under --sample on small subsets (full-benchmark videos always
    # have hundreds of visible points). Return NaN so np.nanmean downstream
    # can drop this video from AD/MD aggregation while OA/OF1 still count it.
    if np.sum(l2_eval_mask) == 0:
        avg_distance = float('nan')
        median_distance = float('nan')
    else:
        avg_distance = np.sum(masked_L2_error) / np.sum(l2_eval_mask)
        assert np.allclose(avg_distance, nonzero_masked_error.mean())
        median_distance = np.median(nonzero_masked_error)
    metrics["avg_distance"] = np.array([avg_distance])
    metrics["median_distance"] = np.array([median_distance])

    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(np.square(pred_tracks - gt_tracks), axis=-1) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        count_correct = np.sum(is_correct & evaluation_points, axis=(1, 2))
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        metrics["num_visible"] = count_visible_points
        metrics["num_pts_within_" + str(thresh)] = count_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(is_correct & pred_visible & evaluation_points, axis=(1, 2))
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)

    metrics["average_jaccard"] = np.mean(np.stack(all_jaccard, axis=1), axis=1)
    metrics["average_pts_within_thresh"] = np.mean(np.stack(all_frac_within, axis=1), axis=1)
    return metrics


def grade(root_dirs, pkl_path, dataset_json_path, occ_thresh=None, sample=False):

    def uid_to_vidname(u):
        return '_'.join(u.split(',')[0].split('_')[:-1])

    with open(pkl_path, 'rb') as f:
        davis = pickle.load(f)
    with open(dataset_json_path, 'r') as f:
        gt_json_dataset = json.load(f)
    expected_counts = Counter(uid_to_vidname(x['uid']) for x in gt_json_dataset)

    all_json_files = []
    for rd in root_dirs:
        all_json_files.extend(glob(os.path.join(rd, '**', 'tapvid*.json'), recursive=True))
    assert all_json_files, f"No tapvid_*.json files under {root_dirs}"
    print(f'Found {len(all_json_files)} prediction shard files. Sample: {all_json_files[0]}')

    results_per_vid = {}
    for json_file in tqdm(all_json_files, desc="loading predictions"):
        with open(json_file, 'r') as f:
            res = json.load(f)
        for item in res:
            uid = item.pop('uid')
            qf, tf, pi = uid.split(',')
            splits = qf.split('_')
            vidname = '_'.join(splits[:-1])
            st = int(splits[-1])
            et = int(tf.split('_')[-1])
            pi = int(pi)
            results_per_vid.setdefault(vidname, {})[(st, et, pi)] = item

    all_results = defaultdict(list)
    expected_total = sum(expected_counts.values())
    actual_total = 0
    for vidname in results_per_vid:
        expec_count = expected_counts[vidname]
        act_count = len(results_per_vid[vidname])
        actual_total += act_count
        print(f"{vidname}: Expected {expec_count}, got {act_count} ({act_count * 100 / expec_count:.3f}%)")

        data = davis[vidname]
        n, t = data['points'].shape[:-1]
        query_points = np.zeros((1, n, 3))
        gt_occluded = np.zeros((1, n, t)); gt_occluded[0] = data['occluded']
        gt_tracks = np.zeros((1, n, t, 2))
        pred_occluded = np.zeros_like(gt_occluded)
        pred_tracks = np.zeros_like(gt_tracks)
        # bool (not float64 as in the original ccwm script) so it composes with
        # the bool masks inside compute_tapvid_metrics when --sample passes it
        # in. Without this --sample raises TypeError on `bool & float64` — a
        # latent bug in the source that fires on every modern numpy.
        evaluation_points = np.zeros((1, n, t), dtype=bool)

        for (st, et, pi), res in results_per_vid[vidname].items():
            query_points[0, pi] = [st, res['gt_query_y'], res['gt_query_x']]
            gt_occluded[0, pi, et] = res['gt_occ']
            gt_tracks[0, pi, et] = [res['gt_target_x'], res['gt_target_y']]
            pred_occluded[0, pi, et] = res['pred_occ']
            if occ_thresh is not None:
                pred_occluded[0, pi, et] = res['occ_metric'] < occ_thresh
            pred_tracks[0, pi, et] = [res['pred_target_x'], res['pred_target_y']]
            evaluation_points[0, pi, et] = True

        tapvid_results = compute_tapvid_metrics(
            query_points=query_points.astype(np.float32),
            gt_occluded=gt_occluded.astype(bool),
            gt_tracks=gt_tracks.astype(np.float32),
            pred_occluded=pred_occluded.astype(bool),
            pred_tracks=pred_tracks.astype(np.float32),
            query_mode='first',
            evaluation_points=evaluation_points if sample else None,
            mask_out_query_point_for_ad=sample,
        )
        for k, v in tapvid_results.items():
            all_results[k].extend(v.tolist())

    print(f'\nAD  (avg distance)            : {np.nanmean(all_results["avg_distance"]):.4f}')
    print(f'MD  (median distance)         : {np.nanmean(all_results["median_distance"]):.4f}')
    print(f'Pct (pixel-threshold accuracy): {np.nanmean(all_results["average_pts_within_thresh"]):.4f}')
    print(f'AJ  (avg jaccard)             : {np.nanmean(all_results["average_jaccard"]):.4f}')
    print(f'OA  (occlusion accuracy)      : {np.nanmean(all_results["occlusion_accuracy"]):.4f}')

    occ_tp = np.sum(all_results["occ_tp"])
    occ_fp = np.sum(all_results["occ_fp"])
    occ_fn = np.sum(all_results["occ_fn"])
    precision = occ_tp / (occ_tp + occ_fp) if (occ_tp + occ_fp) > 0 else float('nan')
    recall = occ_tp / (occ_tp + occ_fn) if (occ_tp + occ_fn) > 0 else float('nan')
    occ_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else float('nan')
    print(f'OF1 (occlusion F1)            : {occ_f1:.4f}')
    print(f'Occlusion Recall              : {recall:.4f}')
    print(f"\nTotal: {actual_total}/{expected_total} finished ({actual_total * 100 / expected_total:.3f}%)")


def main():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Comma-separated dirs holding eval_tapvid_flow output (with results/tapvid_*.json).')
    parser.add_argument('--pkl_path', type=str, required=True,
                        help='Path to the official TapVID-DAVIS pickle (for ground truth tracks).')
    parser.add_argument('--data_path', type=str, default='data/evals/flow/tapvid_davis_first/dataset.json',
                        help='Path to the bundled dataset.json (for expected per-video counts).')
    parser.add_argument('--occ_thresh', type=float, default=None,
                        help='If set, override pred_occ with (occ_metric < occ_thresh).')
    parser.add_argument('--sample', action='store_true',
                        help='Restrict evaluation to the points actually present in the prediction JSONs.')
    args = parser.parse_args()

    root_dirs = args.root_dir.split(',')
    grade(root_dirs, args.pkl_path, args.data_path, args.occ_thresh, args.sample)


if __name__ == '__main__':
    main()
