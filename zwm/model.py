"""
Zero-shot World Model (ZWM) implementation

"""

import math
import inspect
from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tqdm
from PIL import Image

from zwm.utils.model_wrapper import WrappedModel
from zwm.utils.modeling import LayerNorm, Block, PSIBlock, RMSNorm



class ZWM(WrappedModel):

    def __init__(self, config):
        super().__init__(config)
        print("using config:", config, flush=True)
        self.config = config

        flattened_patch_size = (config.patch_size ** 2) * config.n_input_channels

        if config.loss_function == 'l1':
            self.loss = F.l1_loss
        elif config.loss_function == 'l2':
            self.loss = F.mse_loss

        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Linear(flattened_patch_size, config.n_embd),
            positional_embedding = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.mask_token_embedding = nn.Parameter(torch.rand(config.n_embd))
        self.patch_head = nn.Linear(config.n_embd, flattened_patch_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # LLM thing, probably not significant
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # set spmd mesh to None by default, set to the actual mesh if using spmd
        self.spmd_mesh = None
        self.unsharded_param_count = self.get_num_params()

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t, c = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # seq: [N, 2048, 192]
        # pos: [N, 2048]
        # mask: [N, 2048]
        # patch_emb: [N, 2048, 768]
        # mask_patch_emb: [N, 2048, 768]
        # pos_emb: [N, 2048, 768]
        # x: [N, 2048, 768]
        patch_emb = self.transformer.token_embedding(seq)
        mask_patch_emb = self.mask_token_embedding.unsqueeze(0).unsqueeze(0).repeat(b, t, 1) * mask.unsqueeze(-1)
        pos_emb = self.transformer.positional_embedding(pos)
        x = self.transformer.drop(patch_emb + pos_emb + mask_patch_emb) # TODO: Make sure to zero out the image in dataloader for masked parts

        if self.spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(x, self.spmd_mesh,  (('dcn', 'data'), None, 'model'))

        for i, block in enumerate(self.transformer.h):
            x = block(x, spmd_mesh=self.spmd_mesh, mask=mask)
        
        x = self.transformer.ln_f(x)

        # if tgt is not none, compute the patch prediction for the entire sequence
        if tgt is None:
            patches = self.patch_head(x)
            return patches, None
        
        # if tgt is not none, compute the patches and the loss for the target sequence
        patches = self.patch_head(x[:, -tgt.size(1):])

        if self.spmd_mesh is not None:
            xs.mark_sharding(patches, self.spmd_mesh, (('dcn', 'data'), None, 'model'))

        # patches: [N, 921, 192]
        # tgt: [N, 921, 192]
        loss = self.loss(patches, tgt)
        return patches, loss

    def forward_and_return_attention(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t, c = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # seq: [N, 2048, 192]
        # pos: [N, 2048]
        # mask: [N, 2048]
        # patch_emb: [N, 2048, 768]
        # mask_patch_emb: [N, 2048, 768]
        # pos_emb: [N, 2048, 768]
        # x: [N, 2048, 768]
        patch_emb = self.transformer.token_embedding(seq)
        mask_patch_emb = self.mask_token_embedding.unsqueeze(0).unsqueeze(0).repeat(b, t, 1) * mask.unsqueeze(-1)
        pos_emb = self.transformer.positional_embedding(pos)
        x = self.transformer.drop(patch_emb + pos_emb + mask_patch_emb) # TODO: Make sure to zero out the image in dataloader for masked parts

        if self.spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(x, self.spmd_mesh,  (('dcn', 'data'), None, 'model'))

        att_list = []
        for i, block in enumerate(self.transformer.h):
            x, att = block.forward_and_return_attention(x, spmd_mesh=self.spmd_mesh, mask=mask)
            att_list.append(att)

        # merge the attention list into a single numpy array
        att_list = torch.cat(att_list, dim=0)  # shape: (N_layers, N_patches, N_patches)
        att_list = att_list.detach().cpu().to(torch.float32).numpy()

        x = self.transformer.ln_f(x)

        # if tgt is not none, compute the patch prediction for the entire sequence
        if tgt is None:
            patches = self.patch_head(x)
            return patches, None, att_list
        
        # if tgt is not none, compute the patches and the loss for the target sequence
        patches = self.patch_head(x[:, -tgt.size(1):])

        if self.spmd_mesh is not None:
            xs.mark_sharding(patches, self.spmd_mesh, (('dcn', 'data'), None, 'model'))

        # patches: [N, 921, 192]
        # tgt: [N, 921, 192]
        loss = self.loss(patches, tgt)
        return patches, loss, att_list

    def forward_and_return_activations(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """

        # grab device to perform operations on
        device = seq.device
        # grab dimensions
        b, t, c = seq.size()

        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # create a tensor of position indices, if not provided
        if pos is None:
            pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)

        # seq: [N, 2048, 192]
        # pos: [N, 2048]
        # mask: [N, 2048]
        # patch_emb: [N, 2048, 768]
        # mask_patch_emb: [N, 2048, 768]
        # pos_emb: [N, 2048, 768]
        # x: [N, 2048, 768]
        patch_emb = self.transformer.token_embedding(seq)
        mask_patch_emb = self.mask_token_embedding.unsqueeze(0).unsqueeze(0).repeat(b, t, 1) * mask.unsqueeze(-1)
        pos_emb = self.transformer.positional_embedding(pos)
        x = self.transformer.drop(patch_emb + pos_emb + mask_patch_emb) # TODO: Make sure to zero out the image in dataloader for masked parts

        if self.spmd_mesh is not None:
            import torch_xla.distributed.spmd.xla_sharding as xs
            xs.mark_sharding(x, self.spmd_mesh,  (('dcn', 'data'), None, 'model'))

            
        activations = []
        for i, block in enumerate(self.transformer.h):
            x = block(x, spmd_mesh=self.spmd_mesh, mask=mask)
            activations.append(x)
        activations = np.concatenate([act.detach().cpu() for act in activations], axis=0)

        x = self.transformer.ln_f(x)

        # if tgt is not none, compute the patch prediction for the entire sequence
        if tgt is None:
            patches = self.patch_head(x)
            return patches, None, activations
        
        # if tgt is not none, compute the patches and the loss for the target sequence
        patches = self.patch_head(x[:, -tgt.size(1):])

        if self.spmd_mesh is not None:
            xs.mark_sharding(patches, self.spmd_mesh, (('dcn', 'data'), None, 'model'))

        # patches: [N, 921, 192]
        # tgt: [N, 921, 192]
        loss = self.loss(patches, tgt)
        return patches, loss, activations

    def estimate_mfu(self, fwdbwd_per_iter, T, dt, gpu_type='A40'):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.unsharded_param_count # self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head
        # L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second

        # grab promised flops based on GPU type
        if gpu_type == 'A40':
            flops_promised = 149.7e12 # A40 GPU bfloat16 peak flops is 149.7 TFLOPS
        elif gpu_type == 'A100':
            flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        elif gpu_type == 'H100':
            flops_promised = 756e12 # H100 GPU bfloat16 peak flops is 756 TFLOPS
        elif gpu_type == 'TPUv4':
            flops_promised = 275e12
        elif gpu_type == 'TPUv5e':
            flops_promised = 197e12

        mfu = flops_achieved / flops_promised
        return mfu


class ZWM2(WrappedModel):

    def __init__(self, config):
        super().__init__(config)
        print("using config:", config, flush=True)
        self.config = config

        flattened_patch_size = (config.patch_size ** 2) * config.n_input_channels

        if config.loss_function == 'l1':
            self.loss = F.l1_loss
        elif config.loss_function == 'l2':
            self.loss = F.mse_loss

        self.transformer = nn.ModuleDict(dict(
            token_embedding = nn.Linear(flattened_patch_size, config.n_embd),
            channel_embedding = nn.Embedding(config.channel_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([PSIBlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, bias=config.bias),
        ))
        
        self.patch_head = nn.Linear(config.n_embd, flattened_patch_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        
        self.unsharded_param_count = self.get_num_params()

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(
            self, 
            seq: torch.Tensor, 
            pos: torch.Tensor = None, 
            tgt: torch.Tensor = None, 
            mask: torch.Tensor = None,
        ) -> torch.Tensor:
        """
        Forward pass of the model

        Parameters:
            seq (torch.Tensor) of size b, t: The input sequence
            pos (torch.Tensor) of size b, t: The positional indices of the sequence
            tgt (torch.Tensor) of size b, t_tgt: The target sequence
            mask (torch.Tensor) of size b, t: The mask of the sequence
        
        Returns:
            torch.Tensor: The logits of the model. Size b, t if tgt is None, else b, t_tgt
        """
        
        st_pos = pos[:, :, :-1] # st: space-time (x,y,t)
        channel_pos = pos[:, :, -1] # -1 is mask dimension

        # forward the GPT model itself
        # self.transformer.token_embedding: Linear(in_features=192, out_features=768, bias=True)
        # seq: [16, 3648, 192]
        # tok_emb: [16, 3648, 768]
        # self.transformer.channel_embedding: Embedding(2, 768)
        # channel_pos: [16, 3648]
        # channel_emb: should be [16, 3648, 768]
        
        tok_emb = self.transformer.token_embedding(seq) # token embeddings of shape (b, t, n_embd)
        channel_emb = self.transformer.channel_embedding(channel_pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + channel_emb)

        for i, block in enumerate(self.transformer.h):
            x = block(x, pos=st_pos)
        
        x = self.transformer.ln_f(x)

        # if tgt is not none, compute the patch prediction for the entire sequence
        if tgt is None:
            patches = self.patch_head(x)
            return patches, None
        
        # if tgt is not none, compute the patches and the loss for the target sequence
        patches = self.patch_head(x[:, -tgt.size(1):])

        # patches: [N, 921, 192]
        # tgt: [N, 921, 192]
        loss = self.loss(patches, tgt)
        return patches, loss

    def estimate_mfu(self, fwdbwd_per_iter, T, dt, gpu_type='A40'):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.unsharded_param_count # self.get_num_params()
        cfg = self.config
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head
        # L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second

        # grab promised flops based on GPU type
        if gpu_type == 'A40':
            flops_promised = 149.7e12 # A40 GPU bfloat16 peak flops is 149.7 TFLOPS
        elif gpu_type == 'A100':
            flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        elif gpu_type == 'H100':
            flops_promised = 756e12 # H100 GPU bfloat16 peak flops is 756 TFLOPS
        elif gpu_type == 'TPUv4':
            flops_promised = 275e12
        elif gpu_type == 'TPUv5e':
            flops_promised = 197e12

        mfu = flops_achieved / flops_promised
        return mfu