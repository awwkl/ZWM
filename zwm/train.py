"""
Training Script for ZWM

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 8 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=8 train.py

To run with FSDP on 2 nodes with 8 gpus each, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=10.102.2.210 --master_port=8001 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=10.102.2.210  --master_port=8001 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

"""


import os
import time
import math
import argparse
import importlib
from contextlib import nullcontext
import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import gc
import psutil
import subprocess
import datetime

from zwm.data.patch_sequence_dataset import PatchSequenceDataset
from zwm.utils.viz import fig_to_img
from zwm.utils.model_wrapper import cfg_to_dict, ModelFactory

from zwm.data.image_processing import create_images_from_patches

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # Logging Settings
    parser.add_argument('--run_name', type=str, default='ZWM_170M_Kinetics', help='wandb run name suffix (appended to model name)')
    parser.add_argument('--wandb', default=False, action='store_true', help='log to wandb')
    parser.add_argument('--wandb_project', type=str, default='zwm', help='wandb project name')
    parser.add_argument('--wandb_org', type=str, default='zwm_org', help='wandb org name')
    parser.add_argument('--eval_interval', type=int, default=100, help='how often to run eval')
    parser.add_argument('--save_interval', type=int, default=5000, help='how often to save')
    parser.add_argument('--log_interval', type=int, default=10, help='how often to log')
    parser.add_argument('--eval_iters', type=int, default=1, help='how many iters to eval for')
    parser.add_argument('--eval_only', default=False, action='store_true', help='if True, script exits after eval')
    parser.add_argument("--save_to_gcloud", default=False, action="store_true", help="Whether to save model to gcloud")
    parser.add_argument("--save_only_master", default=False, action="store_true", help="Whether to save model only to master")

    # Training Settings
    parser.add_argument('--resume_from', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--resume_optimizer', default=False, action='store_true', help='resume optimizer from checkpoint')
    parser.add_argument('--batch_size', type=int, default=512, help='cumulative batch size per iteration')
    parser.add_argument('--per_device_batch_size', type=int, default=1, help='batch size per device')
    parser.add_argument('--compile', default=False, action='store_true', help='compile the model')
    parser.add_argument('--device', type=str, default='cpu', help='device to run on')
    parser.add_argument('--backend', type=str, default='nccl', help='ddp backend')
    parser.add_argument('--dtype', type=str, 
        default='bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16', 
        help='data type to use')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--debug', default=False, action='store_true', help='debug mode')
    parser.add_argument('--train_data_dir', type=str, default=None, nargs='+', help='paths to training data')
    parser.add_argument('--train_data_ratio', type=int, default=None, nargs='+', help='ratio of training data to use')
    parser.add_argument('--val_data_dir', type=str, help='path to validation data')
    parser.add_argument('--cache_data', default=False, action='store_true', help='cache the data in memory')
    parser.add_argument('--xla_model_axis', type=int, default=1, help='model axis for xla')
    parser.add_argument('--accelerator_type', type=str, default='H100', help='accelerator type')
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true', help='use gradient checkpointing')
    # parser.add_argument('--dataloader_mode', type=str, default='rgb_shuffle', help='dataloader mode')
    parser.add_argument('--dataloader_mode', type=str, default='zwm_rgb_256', help='dataloader mode')
    parser.add_argument('--eval_dataloader_mode', type=str, default=None, help='dataloader mode for eval')
    parser.add_argument('--model_config', type=str, default='zwm.config.ZWM_170MConfig', help='model config class')
    parser.add_argument('--max_seq_len', type=int, default=None, help='maximum sequence length')
    parser.add_argument('--batches_per_execution', type=int, default=1, help='batches per execution')
    parser.add_argument('--frame0_sparsity', type=float, default=0.0, help='mask ratio for frame 0')
    parser.add_argument('--frame1_sparsity', type=float, default=0.0, help='mask ratio for frame 1')
    parser.add_argument('--frame0_mask_ratio', type=float, default=0.0, help='mask ratio for frame 0')
    parser.add_argument('--frame1_mask_ratio', type=float, default=0.9, help='mask ratio for frame 1')
    parser.add_argument('--num_flow_patches', type=int, default=768, help='number of flow patches to be used as conditioning')
    parser.add_argument('--exotic_mask', type=str, default=None, help='exotic mask')
    parser.add_argument('--fsdp', default=False, action='store_true', help='use fsdp')

    # Optimizer Settings
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='max learning rate')
    parser.add_argument('--max_iters', type=int, default=200001, help='total number of training iterations')
    parser.add_argument('--weight_decay', type=float, default=1e-1, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='adamw beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='adamw beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')

    # Schedules
    parser.add_argument('--decay_lr', type=bool, default=True, help='whether to decay the learning rate')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='how many steps to warm up for')
    parser.add_argument('--warmdown_iters', type=int, default=0, help='how many steps to warm down for')
    parser.add_argument('--lr_decay_iters', type=int, default=200000, help='should be ~= max_iters per Chinchilla')
    parser.add_argument('--start_iter', type=int, default=None, help='starting iteration override, if not set will defualt to 0 or resume from checkpoint')
    parser.add_argument('--min_lr', type=float, default=0, help='minimum learning rate, should be ~= learning_rate/10 per Chinchilla')
    parser.add_argument('--decay_type', type=str, default='cosine', help='cosine or hold')
    
    # Campose Settings
    parser.add_argument('--egomotion_mask_ratio', type=float, default=0.0, help='mask ratio for egomotion')
    parser.add_argument('--prob_egomotion_condition', type=float, default=0.5, help='probability of conditioning on egomotion')
    parser.add_argument('--prob_egomotion_prediction', type=float, default=0.5, help='probability of predicting egomotion')
    parser.add_argument('--campose_cache_path', type=str, default=None, help='path to campose cache')
    
    return parser.parse_args()


def main(args):

    # ----------------------------------------------------------------------------------------------------- #
    ### TRAINING ENVIRONMENT

    # Initialize the ddp, xla or single thread training environment
    ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    gradient_accumulation_steps = args.batch_size // args.per_device_batch_size
    if ddp and not args.debug:
        init_process_group(backend=args.backend, timeout=datetime.timedelta(seconds=3000))
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    elif args.device == 'xla':
        # when using spmd, the entire script gets run as if it was a single program
        # so we do not need to distinguish between master and worker processes
        master_process = xr.process_index() == 0
        seed_offset = 0
        ddp_world_size = xr.global_runtime_device_count()
        ddp_local_rank = xm.get_local_ordinal()
        # using SPDM we pretend that the entire TPU pod is a single device hence we need
        # to divide the desired global batch size with the "per device" batch size
        # to get the gradient accumulation steps
        gradient_accumulation_steps = args.batch_size // args.per_device_batch_size
        # If we are using SPDM the program pretends that the whole TPU pod is a single device
        # so we set world size to 1 and the rank to 0
        ddp_rank = 0
        ddp_world_size = 1
        device = xm.xla_device()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        device = args.device

    # Set seeds for reproducibility
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    # Define the model context depending on device and data type
    device_type = 'cpu'
    ctx = nullcontext()
    if 'cuda' in args.device:
        device_type = 'cuda'
        ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    elif args.device == 'xla':
        device_type = 'xla'
        ctx = torch.autocast(device_type=device_type, dtype=ptdtype)
        # ctx = nullcontext()
        # ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # ----------------------------------------------------------------------------------------------------- #
    ### MODEL AND OPTIMIZER

    # Grab the model config class
    module_path, class_name = args.model_config.rsplit('.', 1)
    module = importlib.import_module(module_path)
    init_cfg_cls = getattr(module, class_name)
    cfg = init_cfg_cls()
    # Grab the model class
    module_path, class_name = cfg.model_class.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_cls = getattr(module, class_name)
    # Set config parameters with command line arguments
    for key, value in vars(args).items():
        if hasattr(cfg, key) and value is not None:
            setattr(cfg, key, value)
    patch_size = cfg.patch_size

    # Initialize model
    model = model_cls(cfg)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # Use this variable to save an optimizer state dict loaded from checkpoint, if there is one
    optimizer_state_dict = None

    # Load model from checkpoint if needed
    if args.resume_from:
        try:
            # this was updated in the previous commit
            logger.info(f"loading model from {args.resume_from}")
            # print(f"loading model from {args.resume_from}", flush=True)
            factory = ModelFactory()
            ckpt_model, ckpt = factory.load_model(args.resume_from, return_ckpt=True, return_optimizer=args.resume_optimizer)

            # # load the model weights, non-strict
            # print("Starting to load the model weights...")
            # model.load_state_dict(
            #     {k.replace("_orig_mod.", ""): w for k, w in ckpt['weights'].items() if k.replace("_orig_mod.", "") not in 
            #     ['lm_head.weight', 'transformer.token_embedding.weight', 'transformer.positional_embedding.weight']}, 
            # strict=False)

            # load the model weights, non-strict
            print("Starting to load the model weights...")
            model.load_state_dict(
                {k.replace("_orig_mod.", "").replace("_orig_module.", ""): w for k, w in ckpt['weights'].items()},
            strict=True)

            new_cfg = cfg_to_dict(cfg)
            # print(f"new config: {new_cfg}")
            old_cfg = ckpt['cfg']
            # print(f"old config: {old_cfg}")
            # get all the keys that has "range" in it
            new_range_keys = [k for k in new_cfg.keys() if 'range' in k]
            new_not_range_keys = [k for k in new_cfg.keys() if 'range' not in k]
            # print(f"new range keys: {new_range_keys}")
            old_range_keys = [k for k in old_cfg.keys() if 'range' in k]
            # print(f"old range keys: {old_range_keys}")
            # get all the overlapping keys
            overlapping_keys = sorted(list(set(new_range_keys) & set(old_range_keys)))
            newly_added_cfg_keys = list(set(new_range_keys) - set(old_range_keys))
            old_deprecated_cfg_keys = list(set(old_range_keys) - set(new_range_keys))

            logger.info("\nCONFIG LOADING SUMMARY:\n")
            logger.info(f"  - Overlapping Range Keys (Loaded): {overlapping_keys if overlapping_keys else 'None'}")
            logger.info(f"  - Newly Added Range Keys (Not Loaded): {newly_added_cfg_keys if newly_added_cfg_keys else 'None'}")
            logger.info(f"  - Deprecated Range Keys (Not Loaded): {old_deprecated_cfg_keys if old_deprecated_cfg_keys else 'None'}\n")
            logger.info(f"  - The Other Config Keys (Not Loaded): {new_not_range_keys if new_not_range_keys else 'None'}\n")

            # if len(overlapping_keys) != 0:
            #     # if 'pos_range' in key, then it should change positional_embedding.weight 
            #     # else it should change lm_head.weight and token_embedding.weight
            #     with torch.no_grad():
            #         for key in overlapping_keys:
            #             if 'pos_range' in key:
            #                 model.transformer.positional_embedding.weight[new_cfg[key][0]:new_cfg[key][1]] = \
            #                     ckpt_model.transformer.positional_embedding.weight[old_cfg[key][0]:old_cfg[key][1]].detach().clone()
            #             else:
            #                 model.lm_head.weight[new_cfg[key][0]:new_cfg[key][1]] = \
            #                     ckpt_model.lm_head.weight[old_cfg[key][0]:old_cfg[key][1]].detach().clone()
            #                 model.transformer.token_embedding.weight[new_cfg[key][0]:new_cfg[key][1]] = \
            #                     ckpt_model.transformer.token_embedding.weight[old_cfg[key][0]:old_cfg[key][1]].detach().clone()
            
            # # Set all layernorm weights with a value of 1.0 to 0.1
            # logger.info("Setting all layernorm weights with a value of 1.0 to 0.0")
            # with torch.no_grad():
            #     for name, param in model.named_parameters():
            #         if len(param.shape) == 1 and param.data.mean() == 1.0:
            #             # logger.info(f"Setting {name} to 0.0")
            #             # param.data.fill_(0.0)
            #             logger.info(f"Setting {name} to 0.1")
            #             param.data.fill_(0.1)

            start_it = gradient_accumulation_steps * ckpt['iteration'] + 1
            wandb_id = None
            best_val_loss = 1e9
            
            # Store the optimizer
            if args.resume_optimizer and 'optimizer' in ckpt:
                optimizer_state_dict = ckpt['optimizer']            

            del ckpt
            del ckpt_model
            gc.collect()
            
        except Exception as e:
            print("Error loading model from checkpoint:", e)
            start_it = 0
            best_val_loss = 1e9
            wandb_id = None
    # Else initialize a new model from scratch
    else:
        start_it = 0
        best_val_loss = 1e9
        wandb_id = None

    # Move model to device
    if args.device == 'xla':
        model = model.to(device)
    else:
        model = model.to(device)
    # gc.collect()

    # override the start iteration if it is set in the command line
    if args.start_iter is not None:
        start_it = args.start_iter

    if ddp and args.fsdp and not args.device == 'xla':
        # wrap model into FSDP container
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # from torch.distributed.fsdp import MixedPrecision
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
        # from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        # model = model = FSDP(
        #     model,
        #     cpu_offload=CPUOffload(offload_params=True),
        #     mixed_precision=MixedPrecision(
        #         param_dtype=torch.bfloat16,  # Use bfloat16 for parameters
        #         reduce_dtype=torch.bfloat16,  # Use bfloat16 for reductions
        #         buffer_dtype=torch.bfloat16  # Use bfloat16 for buffers
        #     ),
        #     use_orig_params=True,
        # )
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, MixedPrecision
        from functools import partial

        # Import your Block class
        from zwm.utils.modeling import Block

        def get_wrapper():
             auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block},  # Include your Block class here
             )
             return auto_wrap_policy

        # Get the customized wrap policy
        auto_wrap_policy = get_wrapper()

        # Wrap your model with FSDP using the custom wrap policy
        model = FSDP(
            model,
            # cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,  # Use bfloat16 for parameters
                reduce_dtype=torch.bfloat16,  # Use bfloat16 for reductions
                buffer_dtype=torch.bfloat16   # Use bfloat16 for buffers
            ),
            use_orig_params=True,
            auto_wrap_policy=auto_wrap_policy,  # Include the custom wrap policy here
            limit_all_gathers=True,
            device_id=torch.cuda.current_device(),
        )

    # compile the model
    if args.compile:
        print("compiling the model... (takes a ~minute)")
        # unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0. Seems to require 2.6

    # Initialize optimizer. 
    # NOTE: This has to be done after model is moved to device. See https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/3
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, 
                                           (args.beta1, args.beta2), device_type)
    
    # Load optimizer from checkpoint if available
    def remove_orig_mod(obj):
        """
        Recursively traverse the object.
        - If obj is a dict, return a new dict with its keys (if strings) and values processed.
        - If obj is a list, return a new list with its items processed.
        - If obj is a string, remove '_orig_mod.' from it.
        - Otherwise, return obj unchanged.
        """
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                # If the key is a string, replace the substring.
                new_key = key.replace("_orig_mod.", "") if isinstance(key, str) else key
                # Recursively process the value.
                new_dict[new_key] = remove_orig_mod(value)
            return new_dict
        elif isinstance(obj, list):
            return [remove_orig_mod(item) for item in obj]
        elif isinstance(obj, str):
            return obj.replace("_orig_mod.", "")
        else:
            return obj

    # if args.resume_from:
    #     factory = ModelFactory()
    #     ckpt_model, ckpt = factory.load_model(args.resume_from, return_ckpt=True)
    #     if 'optimizer' in ckpt:
    #         logger.info(f"loading optimizer checkpoint from {args.resume_from}")
    #         # remove _orig_mod. from the optimizer state_dict keys
    #         ckpt['optimizer'] = remove_orig_mod(ckpt['optimizer'])

    # optimizer_state_dict
    # {'state': {}, 'param_groups': [{'weight_decay': 0.1, 'lr': 0.0003, 'betas': (0.9, 0.95), 'eps': 1e-08, 'amsgrad': False, 'foreach': None, 
    # 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True, 
    # 'params': [0, 1, ..., 218, 219]}]}
    if args.resume_from and args.resume_optimizer and optimizer_state_dict is not None:
        logger.info(f"loading optimizer checkpoint from {args.resume_from}")

        optimizer_state_dict = remove_orig_mod(optimizer_state_dict)

        param_names = {id(p): name for name, p in model.named_parameters()}
        for group in optimizer.param_groups:
            param_names_in_group = [param_names[id(p)] for p in group['params']]
            print(param_names_in_group)
        if args.fsdp:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            logger.info(f"loading optimizer checkpoint with FSDP state dict")
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                optimizer.load_state_dict(optimizer_state_dict)
            logger.info(f"FSDP - Loaded optimizer done with FullStateDictConfig, StateDictType")

        else:
            optimizer.load_state_dict(optimizer_state_dict)
            logger.info("Loading optimizer done: optimizer.load_state_dict(optimizer_state_dict)")


    # If we are not using xla and gradient checkpointing is on, wrap the layer moduels
    if args.device != 'xla' and args.gradient_checkpointing:
        from zwm.utils.modeling import CheckpointWrapper
        for i, block in enumerate(model.transformer.h):
            model.transformer.h[i] = CheckpointWrapper(block)

    # wrap model into DDP container
    if ddp and not args.fsdp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)

    # if args.device == 'xla':
    #     # If we are using xla, randezvous
    #     xm.rendezvous('resume_checkpoint')

    # ----------------------------------------------------------------------------------------------------- #
    ### DATASET 

    # Define normal dataset if we are using GPU with a distributed sampler
    if args.device != 'xla':
        # define data loader
        train_dataset = PatchSequenceDataset(paths=args.train_data_dir, model_config=cfg,
                                        num_folds=1, fold_idx=0, mode=args.dataloader_mode, 
                                        frame0_sparsity=args.frame0_sparsity, frame1_sparsity=args.frame1_sparsity, 
                                        frame0_mask_ratio=args.frame0_mask_ratio, frame1_mask_ratio=args.frame1_mask_ratio,
                                        path_ratios=args.train_data_ratio, debug=args.debug, max_seq_len=args.max_seq_len, campose_cache_path=args.campose_cache_path,
                                        egomotion_mask_ratio=args.egomotion_mask_ratio,
                                        prob_egomotion_condition=args.prob_egomotion_condition, prob_egomotion_prediction=args.prob_egomotion_prediction)
        
        # Set up the distributed sampler on the trian dataset if using DDP
        sampler = DistributedSampler(
            train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, seed=args.seed
        ) if ddp else None

        train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_batch_size,
                                      prefetch_factor=4,
                                      pin_memory=True, multiprocessing_context='spawn',
                                    sampler=sampler, num_workers=args.num_workers, drop_last=True)
        # if we are not running on xla and are in the master process, make a val dataset as well
        if master_process and args.val_data_dir is not None:
            val_dataset = PatchSequenceDataset(paths=args.val_data_dir, model_config=cfg,
                                        num_folds=1, fold_idx=0, mode=args.dataloader_mode,
                                        frame0_sparsity=args.frame0_sparsity, frame1_sparsity=args.frame1_sparsity, frame1_mask_ratio=args.frame1_mask_ratio,
                                        debug=args.debug)
            val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, 
                                        shuffle=True,  num_workers=args.num_workers, drop_last=True)
        
        tokens_per_iter = args.batch_size * train_dataset.T
        print(f"tokens per iteration will be: {tokens_per_iter:,}, sequence length: {train_dataset.T}")

        if args.num_flow_patches > 0 and args.exotic_mask is not None:
            assert args.exotic_mask == 'blockwise_parallel_flow', f'exotic mask for flow must be blockwise_parallel_flow, but got {args.exotic_mask}'
            args.exotic_mask = f'{args.exotic_mask}_{args.num_flow_patches}'
            print(f'using exotic mask {args.exotic_mask}')

    # shard model across devices
    if args.device == 'xla':


        if args.fsdp:

            print("using FSDP on XLA")

            # grab the number of devices available
            num_devices = xr.global_runtime_device_count()
            device_ids = np.arange(num_devices)

            # the ici mesh 
            mesh_shape = (num_devices, 1)
            mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
            xs.set_global_mesh(mesh)
            # model.spmd_mesh = spmd_mesh

            # # Define the mesh for parameters (all devices in one mesh)
            # param_device_ids = np.arange(256)  # Assuming 256 devices
            # param_mesh_shape = (1, 256)  # Single dimension for parameter mesh
            # param_mesh = xs.Mesh(param_device_ids, param_mesh_shape, ('fsdp', 'data'))

            # # Define the mesh for data (can reuse the same devices but ensure data handling is independent)
            # data_device_ids = np.arange(256)  # Same devices
            # data_mesh_shape = (256, 1)  # Separate dimension for data
            # data_mesh = xs.Mesh(data_device_ids, data_mesh_shape, ('data', 'fsdp'))


            # reduce num accum steps by num devices
            gradient_accumulation_steps //= num_devices

            if master_process:
                print(f"spmd mesh shape: {spmd_mesh}")

            # Shart the parameters of the model
            from torch_xla.experimental.spmd_fully_sharded_data_parallel import SpmdFullyShardedDataParallel as FSDPv2
            from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from functools import partial

            # Import your Block class
            from zwm.utils.modeling import Block

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block},
            )
            model = FSDPv2(
                model, auto_wrap_policy=auto_wrap_policy)

            # for i, block in enumerate(model.transformer.h):
            #     xs.apply_backward_optimization_barrier(model.transformer.h[i])
        
            if args.gradient_checkpointing:
                from torch_xla.distributed.fsdp import checkpoint_module
                for i, block in enumerate(model.transformer.h):
                    model.transformer.h[i] = checkpoint_module(block)

            # define data loader
            train_dataset = PatchSequenceDataset(paths=args.train_data_dir, model_config=cfg,
                                            num_folds=1, fold_idx=0, mode=args.dataloader_mode, 
                                            frame0_sparsity=args.frame0_sparsity, frame1_sparsity=args.frame1_sparsity, frame1_mask_ratio=args.frame1_mask_ratio,
                                            path_ratios=args.train_data_ratio, debug=args.debug, max_seq_len=args.max_seq_len, campose_cache_path=args.campose_cache_path,
                                            egomotion_mask_ratio=args.egomotion_mask_ratio,
                                            prob_egomotion_condition=args.prob_egomotion_condition, prob_egomotion_prediction=args.prob_egomotion_prediction)
            
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size // gradient_accumulation_steps,
                                        shuffle=True, num_workers=args.num_workers, drop_last=True)
        
            train_dataloader = pl.MpDeviceLoader(
                train_dataloader, 
                device,
                # batches_per_execution=args.batches_per_execution,
                input_sharding=xs.ShardingSpec(mesh, ('fsdp', None)),
                loader_prefetch_size=8,#args.per_device_batch_size, 
                device_prefetch_size=2,
            )

            gradient_accumulation_steps = 2
            print("NUMBER OF ACCUMULATION STEPS: ", gradient_accumulation_steps)

        else:

            # grab the number of devices available
            num_devices = xr.global_runtime_device_count()
            device_ids = np.arange(num_devices)
            # dcn axis is "data center network" axis which defines the global per-micro-step
            # batch size ?
            dcn_axis = 1
            # model axis defines the model parallelism
            model_axis = args.xla_model_axis
            # the data axis defines the local per-micro-step batch size
            data_axis = num_devices // model_axis // dcn_axis
            # the ici mesh 
            ici_mesh_shape = (1, data_axis, model_axis)
            spmd_mesh = xs.Mesh(device_ids, ici_mesh_shape, ('dcn', 'data', 'model'))
            xs.set_global_mesh(spmd_mesh)
            model.spmd_mesh = spmd_mesh

            if master_process:
                print(f"spmd mesh shape: {ici_mesh_shape}")

            # Shart the parameters of the model

            for name, layer in model.named_modules():

                # Ignore layers which do not have a weight attribute
                if not hasattr(layer, 'weight') or layer.weight is None:
                    continue

                # Ignore 1D layers (norms, biases, etc)
                if len(layer.weight.shape) == 1:
                    continue

                # Shard all 2D layers
                if len(layer.weight.shape) == 2:

                    if 'fc' in name:
                        if master_process:
                            print(f'layer name {name} weight shape {layer.weight.shape}', flush=True)
                        xs.mark_sharding(layer.weight, spmd_mesh, ('model', 'data'))
                    elif 'mlp.c_proj' in name:
                        if master_process:
                            print(f'layer name {name} weight shape {layer.weight.shape}', flush=True)
                        xs.mark_sharding(layer.weight, spmd_mesh, ('data', 'model'))
                    elif 'attn.c_proj' in name:
                        if master_process:
                            print(f'layer name {name} weight shape {layer.weight.shape}', flush=True)
                        xs.mark_sharding(layer.weight, spmd_mesh, ('model', 'data'))
                    elif 'attn' in name:
                        if master_process:
                            print(f'layer name {name} weight shape {layer.weight.shape}', flush=True)
                        xs.mark_sharding(layer.weight, spmd_mesh, ('data', 'model'))
                    elif 'embedding' in name:
                        if master_process:
                            print(f'layer name {name} weight shape {layer.weight.shape}', flush=True)
                        xs.mark_sharding(layer.weight, spmd_mesh, ('model', 'data'))
                    else:
                        xs.mark_sharding(layer.weight, spmd_mesh, ('model', 'data'))


            for i, block in enumerate(model.transformer.h):
                xs.apply_backward_optimization_barrier(model.transformer.h[i])
        
            if args.gradient_checkpointing:
                from torch_xla.distributed.fsdp import checkpoint_module
                for i, block in enumerate(model.transformer.h):
                    model.transformer.h[i] = checkpoint_module(block)

            # define data loader
            train_dataset = PatchSequenceDataset(paths=args.train_data_dir, model_config=cfg,
                                            num_folds=1, fold_idx=0, mode=args.dataloader_mode, 
                                            frame0_sparsity=args.frame0_sparsity, frame1_sparsity=args.frame1_sparsity, frame1_mask_ratio=args.frame1_mask_ratio,
                                            path_ratios=args.train_data_ratio, debug=args.debug, max_seq_len=args.max_seq_len, campose_cache_path=args.campose_cache_path,
                                            egomotion_mask_ratio=args.egomotion_mask_ratio,
                                            prob_egomotion_condition=args.prob_egomotion_condition, prob_egomotion_prediction=args.prob_egomotion_prediction)
            
            train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_batch_size,
                                        shuffle=True, num_workers=args.num_workers, drop_last=True)
        
            train_dataloader = pl.MpDeviceLoader(
                train_dataloader, 
                device,
                batches_per_execution=args.batches_per_execution,
                input_sharding=xs.ShardingSpec(spmd_mesh, (('dcn', 'data'), None)),
                loader_prefetch_size=8,#args.per_device_batch_size, 
                device_prefetch_size=2,
            )


    # ----------------------------------------------------------------------------------------------------- #
    ### LOGGING

    # Create output path
    if master_process or args.device == 'xla':
        run_name = args.run_name
        out_dir = f'out/{run_name}'
        os.makedirs(out_dir, exist_ok=True)

    # Setup wandb logging
    if args.wandb and master_process:
        wandb.init(project=args.wandb_project, entity=args.wandb_org, 
                   name=run_name, config=vars(args), id=wandb_id)
        # define custom x axis to count step value
        wandb.define_metric("iter")
        # define which metrics will be plotted against it
        wandb.define_metric("*", step_metric="iter")


    # ----------------------------------------------------------------------------------------------------- #
    ### EVAL LOOP
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        # raw_model = model.module if ddp else model
        model.eval()
        for split in ['train', 'val']:
            losses = []
            iter_counter = 0
            print(f"running eval on split {split}")
            for seq, pos, tgt, mask in (train_dataloader if split == 'train' else val_dataloader):
                start_time_eval = time.time()
                # print(f'[iter {iter_counter}] start_time:', start_time_eval, flush=True)
                if 'cuda' in args.device:
                    seq = seq.pin_memory().to(device, non_blocking=True)
                    pos = pos.pin_memory().to(device, non_blocking=True)
                    tgt = tgt.pin_memory().to(device, non_blocking=True)
                    mask = mask.pin_memory().to(device, non_blocking=True)
                # print(f'[iter {iter_counter}] after moving to device:', time.time() - start_time_eval, flush=True)
                with ctx:
                    patches, loss = model(seq, pos, tgt=tgt, mask=mask)
                    if 'cuda' in args.device:
                        torch.cuda.synchronize()
                losses.append(loss.item())
                iter_counter += 1
                if iter_counter >= args.eval_iters or args.debug:
                    break
                # print(f'[iter {iter_counter}] time taken for iter {iter_counter} in split {split}:', time.time() - start_time_eval, flush=True)
            out[f"{split}/loss"] = np.mean(losses)

            if split in ['train', 'val']:
                # Visualize the predictions
                seq, pos, tgt, mask, patches = seq[0].unsqueeze(0), pos[0].unsqueeze(0), tgt[0].unsqueeze(0), mask[0].unsqueeze(0), patches[0].unsqueeze(0)
                frame0, frame1, frame1_pred, frame1_with_mask = create_images_from_patches(seq, pos, tgt, mask, patches, patch_size)
                
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                for axis in ax:
                    axis.title.set_fontsize(35)
                ax[0].imshow(frame0)
                ax[0].set_title("Frame 0")
                ax[1].imshow(frame1)
                ax[1].set_title("Frame 1")
                ax[2].imshow(frame1_pred)
                ax[2].set_title("Prediction")
                ax[3].imshow(frame1_with_mask)
                ax[3].set_title("Mask")

                img = fig_to_img(fig)
                out[f"{split}/factual_prediction"] = img

        model.train()
        return out
    

    # ----------------------------------------------------------------------------------------------------- #
    ### SCHEDULER

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        # 2) if it > warmdown_iters, linearly decay to min learning rate
        if it > args.max_iters - args.warmdown_iters:
            return args.learning_rate - (args.learning_rate - args.min_lr
                ) * (it - (args.max_iters - args.warmdown_iters)) / args.warmdown_iters
        # 4) in between, use cosine decay down to min learning rate
        if args.decay_type == "hold":
            return args.learning_rate
        elif args.decay_type == "cosine":
            decay_ratio = (it - args.warmup_iters) / (args.warmdown_iters - args.warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
            return args.min_lr + coeff * (args.learning_rate - args.min_lr)


    # ----------------------------------------------------------------------------------------------------- #
    ### TRAINING LOOP
    

    # Training loop
    t0 = time.time()
    running_mfu = -1.0
    # running_train_loss = 10.0 # initialize to a high value
    it = start_it

    # If we are not training on TPU we aggregate the loss
    if master_process and args.device != 'xla':
        all_losses = []

    while True:

        for seq, pos, tgt, mask in train_dataloader:

            
            # print(f"SEQ SHAPE: {seq.shape} | POS SHAPE: {pos.shape} | TGT SHAPE: {tgt.shape} | MASK SHAPE: {mask.shape} | SEQ[10:15]: {seq[0, 10:15]}", flush=True)

            # If we are training on GPUs, we need to move the data to the device
            if 'cuda' in args.device:
                seq = seq.pin_memory().to(device, non_blocking=True)
                pos = pos.pin_memory().to(device, non_blocking=True)
                tgt = tgt.pin_memory().to(device, non_blocking=True)
                mask = mask.pin_memory().to(device, non_blocking=True)

            # determine and set the learning rate for this iteration
            lr = get_lr((it // gradient_accumulation_steps)) if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # # evaluate the loss on train/val sets and write checkpoints
            if args.val_data_dir is not None and it % (args.eval_interval * gradient_accumulation_steps) == 0 and master_process:
                losses = estimate_loss()
                print(f"step {it // gradient_accumulation_steps}: train loss {losses['train/loss']:.4f}, val loss {losses['val/loss']:.4f}")
                if args.wandb:
                    wandb.log({
                        # "val_on_train/iter": it // gradient_accumulation_steps,
                        # "val/iter": it // gradient_accumulation_steps,
                        "val_on_train/loss": losses['train/loss'],
                        "val/loss": losses['val/loss'],
                        "val_on_train/prediction": wandb.Image(losses['train/factual_prediction']),
                        "val/prediction": wandb.Image(losses['val/factual_prediction']),
                        # "lr": lr,
                        # "mfu": running_mfu * 100,  # convert to percentage
                    }, step=it // gradient_accumulation_steps)
                # if losses['val/loss'] < best_val_loss:
                #     raw_model = model.module if ddp else model
                #     best_val_loss = losses['val/loss']
                #     raw_model.save(
                #         {'weights': raw_model.state_dict(), 'iteration': it // gradient_accumulation_steps,
                #         'best_val_loss': best_val_loss, 'cfg': cfg_to_dict(cfg), 'args': args,
                #         'wandb_id': wandb.run.id if args.wandb else None}, 
                #         os.path.join(out_dir, "model_best.pt"), gcloud=args.save_to_gcloud)
            if args.eval_only and it == 0:
                break
        
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (it % gradient_accumulation_steps == 0)

            with ctx:
                _, loss = model(seq, pos, tgt=tgt, mask=mask)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation

            # backward pass, with gradient scaling if training in fp16
            if args.device == 'xla':
                loss.backward()
            else:
                scaler.scale(loss).backward()

            if it % gradient_accumulation_steps == 0:
                # clip the gradient
                if args.grad_clip != 0.0:
                    if args.device != 'xla':
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                # step the optimizer and scaler if training in fp16

                # set xm barrier for xla
                if args.device == 'xla':
                    optimizer.step()
                    xm.mark_step()
                else:
                    scaler.step(optimizer)
                    scaler.update()


                # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                # # Synchronize and grab the weights and gradients of all layer norms in the model
                # layer_norms = [p for n, p in model.named_parameters() if len(p.shape) == 1]

                # # Access parameters and gradients using FSDP
                # layer_norm_weights = []
                # layer_norm_grads = []

                # for n, param in model.named_parameters():
                #     if "ln" in n and len(param.shape) == 1:  # Filter LayerNorm parameters
                #         with FSDP.summon_full_params(model, writeback=False):
                #             # Move parameters to CPU to avoid OOM on GPU
                #             layer_norm_weights.append(param.data.cpu().clone())
                #             if param.grad is not None:
                #                 layer_norm_grads.append(param.grad.data.cpu().clone())

                # # Ensure all parameters and gradients are gathered
                # if layer_norm_grads:
                #     all_grads = torch.cat([p.view(-1) for p in layer_norm_grads])
                #     min_grad, max_grad = all_grads.min().item(), all_grads.max().item()
                # else:
                #     min_grad, max_grad = None, None

                # if layer_norm_weights:
                #     all_weights = torch.cat([p.view(-1) for p in layer_norm_weights])
                #     min_weight, max_weight = all_weights.min().item(), all_weights.max().item()
                # else:
                #     min_weight, max_weight = None, None

                # # Log the min and max of the gradients and weights
                # print(f"min_grad: {min_grad}, max_grad: {max_grad}, min_weight: {min_weight}, max_weight: {max_weight}")
                # print("-----")

                    # # grab the weitghts and gradients of all layer norms in the model
                    # layer_norms = [p for n, p in model.named_parameters() if len(p.shape) == 1]
                    # layer_norm_grads = [p.grad for p in layer_norms]
                    # layer_norm_weights = [p for p in layer_norms]
                    
                    # # grab min and max of the gradients and weights across all layer norms
                    # min_grad = torch.cat([p.view(-1) for p in layer_norm_grads]).min()
                    # max_grad = torch.cat([p.view(-1) for p in layer_norm_grads]).max()
                    # min_weight = torch.cat([p.view(-1) for p in layer_norm_weights]).min()
                    # max_weight = torch.cat([p.view(-1) for p in layer_norm_weights]).max()

                    # # log the min and max of the gradients and weights
                    # print(f"min_grad: {min_grad}, max_grad: {max_grad}, min_weight: {min_weight}, max_weight: {max_weight}")
                    # print("-----")
                
                # flush the gradients as soon as we can, no need for this memory anymore
                # optimizer.zero_grad()
                optimizer.zero_grad(set_to_none=True)

            # If we are not training on TPU we need to append the loss to the list
            if master_process and args.device != 'xla':
                all_losses.append(loss.detach().cpu().item() * gradient_accumulation_steps)

            if it % (args.save_interval * gradient_accumulation_steps) == 0 and it != 0 and it != start_it:
                # If we are training on xla, we need to save the model in a different way
                # on every worker
                if not args.save_only_master:
                    if args.device == 'xla':
                        xm.mark_step()
                        xm.rendezvous('start_checkpointing')
                        gc.collect()
                        # log worker id and CPU memory usage:
                        memory_info = psutil.virtual_memory()
                        available_memory_gb = memory_info.available / (1024 ** 3)
                        for f in os.listdir(out_dir):
                            if f.startswith('model_') and f.endswith('.pt'):
                                os.remove(os.path.join(out_dir, f))
                        logger.info(f"worker {xr.process_index()} memory {available_memory_gb:.2f}GB")
                        xm.save(
                            {'weights': model.state_dict(), 'iteration': it // gradient_accumulation_steps,
                            'cfg': cfg_to_dict(cfg), 'args': args},
                        os.path.join(out_dir, f"model_{it//gradient_accumulation_steps:08}.pt"))
                        xm.rendezvous('saved_model')
                        gc.collect()
                        xm.rendezvous('done_checkpointing')
                        if master_process:
                            # Upload the model to gcloud
                            model_path = os.path.join(out_dir, f"model_{it//gradient_accumulation_steps:08}.pt")
                            upload_path = f"gs://zwm/zwm/models/{args.run_name}/model_{it//gradient_accumulation_steps:08}.pt"
                            command = ["gsutil", "cp", model_path, upload_path]
                            subprocess.run(command)
                            logger.info(f"uploaded model to gcloud ☁️ ")

                    if args.device != 'xla':
                        if args.fsdp:
                            # Save FSDP model
                            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                            # Ensure the FSDP model is flattened to CPU
                            from torch.distributed.fsdp import StateDictType
                            from torch.distributed.fsdp import FullStateDictConfig
                            from torch.distributed.fsdp import FullOptimStateDictConfig

                            # save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                            # with FSDP.state_dict_type(
                            #     model, StateDictType.FULL_STATE_DICT, save_policy
                            #     ):
                            #     cpu_state = model.state_dict()
                            #     cpu_optimizer_state = optimizer.state_dict()

                            # Configuration for saving the model state dict:
                            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                            # Configuration for saving the optimizer state dict:
                            # optim_state_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

                            # Save the model's full state dict under FSDP:    
                            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                                cpu_state = model.state_dict()

                            # Save the optimizer's full state dict under FSDP:
                            # with FSDP.full_optim_state_dict(model, optimizer, optim_state_config) as full_optim_state:
                            # with FSDP.optim_state_dict(model, optimizer) as optim_state:
                            #     cpu_optimizer_state = optim_state

                            # with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                            #     cpu_optimizer_state = FSDP.optim_state_dict(model, optimizer)


                            if master_process:

                                # Save the model checkpoint
                                raw_model = model.module if ddp else model
                                raw_model.save(
                                    {
                                        'weights': cpu_state,
                                        'iteration': it // gradient_accumulation_steps,
                                        'cfg': cfg_to_dict(cfg),
                                        'args': args,
                                        'wandb_id': wandb.run.id if args.wandb else None
                                    },
                                    os.path.join(out_dir, f"model_{it // gradient_accumulation_steps:08}.pt")
                                )

                        elif not args.fsdp and master_process:
                            # Save DDP or non-distributed model
                            raw_model = model.module if ddp else model
                            raw_state = {k: v.cpu().float() for k, v in raw_model.state_dict().items()}
                            raw_model.save(
                                {
                                    'weights': raw_state, 
                                    # 'optimizer': optimizer.state_dict(),
                                    'iteration': it // gradient_accumulation_steps,
                                    'cfg': cfg_to_dict(cfg),
                                    'args': args,
                                    'wandb_id': wandb.run.id if args.wandb else None
                                },
                                os.path.join(out_dir, f"model_{it // gradient_accumulation_steps:08}.pt")
                            )

                        # Move the model back to the device
                        # model.to(device)

                    # If we are training on xla, we need to remove the model from the device
                    if args.device == 'xla':
                        xm.mark_step()
                        xm.rendezvous('done_checkpointing')
                        gc.collect()
                        os.remove(os.path.join(out_dir, f"model_{it//gradient_accumulation_steps:08}.pt"))
                else:
                    if args.device == 'xla':
                        xm.mark_step()
                        xm.rendezvous('start_checkpointing')
                        # gc.collect()

                        # Log worker ID and CPU memory usage
                        memory_info = psutil.virtual_memory()
                        available_memory_gb = memory_info.available / (1024 ** 3)
                        logger.info(f"worker {xm.get_ordinal()} memory {available_memory_gb:.2f}GB")

                        # fourth trial
                        cpu_model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                    
                    if master_process:
                        if args.fsdp:
                            try:
                                logger.info("FSDP Sharded Master process saving model to local...")
                                model_path = os.path.join(out_dir, f"model_{it//gradient_accumulation_steps:08}.pt")

                                import torch.distributed.checkpoint as dist_cp
                                import torch_xla.experimental.distributed_checkpoint as xc

                                # Saving a state_dict
                                state_dict = {
                                    "model": model.state_dict(),
                                    # "optimizer": optimizer.state_dict(),
                                    'iteration': it // gradient_accumulation_steps,
                                    'cfg': cfg_to_dict(cfg), 'args': args,
                                }

                                dist_cp.save(
                                    state_dict=state_dict,
                                    storage_writer=dist_cp.FileSystemWriter(model_path),
                                    planner=xc.SPMDSavePlanner(),
                                )

                                logger.info(f"Saved model to local at {model_path} ☁️ ")
                                logger.info("Master process uploading model to GCS...")
                                upload_path = f"gs://zwm/zwm/models/{args.run_name}/model_{it//gradient_accumulation_steps:08}.pt"
                                command = ["gsutil", "cp", model_path, upload_path]
                                subprocess.run(command)
                                logger.info(f"uploaded model to gcloud ☁️ ")
                                os.remove(model_path)
                                logger.info(f"removed model from local at {model_path} ☁️ ")

                            except Exception as e:
                                logger.error(f"Failed to upload model to GCS: {str(e)}")
                        else:
                            try:
                                logger.info("Master process saving model to local...")
                                model_path = os.path.join(out_dir, f"model_{it//gradient_accumulation_steps:08}.pt")
                                torch.save(
                                    {'weights': cpu_model_state_dict, 'iteration': it // gradient_accumulation_steps,
                                    'cfg': cfg_to_dict(cfg), 'args': args},
                                    model_path
                                )
                                logger.info(f"Saved model to local at {model_path} ☁️ ")
                                logger.info("Master process uploading model to GCS...")
                                upload_path = f"gs://zwm/zwm/models/{args.run_name}/model_{it//gradient_accumulation_steps:08}.pt"
                                command = ["gsutil", "cp", model_path, upload_path]
                                subprocess.run(command)
                                logger.info(f"uploaded model to gcloud ☁️ ")
                                os.remove(model_path)
                                logger.info(f"removed model from local at {model_path} ☁️ ")

                            except Exception as e:
                                logger.error(f"Failed to upload model to GCS: {str(e)}")
                    else:
                        # sleep for a bit to allow the master process to finish uploading
                        time.sleep(600)
                    
                    # Sync all workers
                    if args.device == 'xla':
                        del cpu_model_state_dict
                        gc.collect()
                        xm.rendezvous('saved_model')

            if it % (args.log_interval * gradient_accumulation_steps) == 0 and master_process:
                # get the time since the last log
                t1 = time.time()
                dt = (t1 - t0) / args.log_interval
                t0 = t1
                # logger.info(f"iter {it // gradient_accumulation_steps} time {dt * 1000:.2f}ms")

                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                if args.device == 'xla':
                    lossf = loss.item() * gradient_accumulation_steps
                else:
                    lossf = np.mean(all_losses[-256:])  # average over last 256 steps
                if it >= 20:  # let the training loop settle a bit
                    raw_model = model.module if ddp else model
                    # grab mfus (if using xla get tpuv5e spec, else get a40 spec)
                    if args.device == 'xla':
                        mfu = raw_model.estimate_mfu(args.batch_size // num_devices, train_dataset.T, dt, gpu_type='TPUv5e')
                    else:
                        mfu = raw_model.estimate_mfu(args.per_device_batch_size * gradient_accumulation_steps,
                                                     train_dataset.T, dt, gpu_type=args.accelerator_type)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

                memory_info = psutil.virtual_memory()
                available_memory_gb = memory_info.available / (1024 ** 3)


                if 'cuda' in args.device:
                    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
                    logger.info(f"iter {it // gradient_accumulation_steps}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                                f"mfu {running_mfu * 100:.2f}%, cpu mem: {available_memory_gb:.2f}GB, "
                                f"gpu mem allocated: {gpu_memory_allocated:.2f}GB, gpu mem reserved: {gpu_memory_reserved:.2f}GB")
                else:
                    logger.info(f"iter {it // gradient_accumulation_steps}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                                f"mfu {running_mfu * 100:.2f}%, cpu mem: {available_memory_gb:.2f}GB")
                    
                if args.wandb:
                    log_data = {
                        "iter": it // gradient_accumulation_steps,
                        "train/loss": lossf,
                        "train/mfu": running_mfu,
                        "cpu memory": available_memory_gb,
                        "lr": lr,
                    }
                    if 'cuda' in args.device:
                        log_data.update({
                            "gpu mem allocated": gpu_memory_allocated,
                            "gpu mem reserved": gpu_memory_reserved
                        })
                    wandb.log(log_data, step=it // gradient_accumulation_steps)
                    wandb.log({}, step=it // gradient_accumulation_steps, commit=True) # Flush changes

            # increment iteration counter
            it += 1

            # check if we have reached the maximum number of iterations
            if it >= (args.max_iters * gradient_accumulation_steps):
                if ddp:
                    destroy_process_group()
                return
        

if __name__ == "__main__":
    args = get_args()
    print(args)

    if args.device == 'xla':
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd.xla_sharding as xs
        xr.use_spmd()

    main(args)
