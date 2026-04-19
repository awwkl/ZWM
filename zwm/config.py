from dataclasses import dataclass, field

from zwm.utils.model_wrapper import BaseConfig

@dataclass
class ZWM_170MConfig(BaseConfig):
    '''
    model = PretrainVisionTransformer(
        img_size=256,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=768,
        decoder_num_heads=12,
        decoder_depth=12,
        mlp_ratio=4,
        qkv_bias=True,
        k_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    '''
    model_class: str = "zwm.model.ZWM"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    n_layer: int = 24
    n_head: int = 12
    n_embd: int = 768
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    causal_attention: bool = False

@dataclass
class ZWM_1BConfig(BaseConfig):
    model_class: str = "zwm.model.ZWM"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    n_layer: int = 48
    n_head: int = 16
    n_embd: int = 1280
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    causal_attention: bool = False

@dataclass
class ZWM_7BConfig(BaseConfig):
    model_class: str = "zwm.model.ZWM"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    n_layer: int = 48
    n_head: int = 32
    n_embd: int = 3456
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    causal_attention: bool = False

@dataclass
class ZWM2_170MConfig(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    channel_size: int = 2
    n_layer: int = 24
    n_head: int = 12
    n_embd: int = 768
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"

@dataclass
class ZWM2_1BConfig(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    channel_size: int = 2
    n_layer: int = 48
    n_head: int = 16
    n_embd: int = 1280
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"

@dataclass
class ZWM2_4BConfig(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    channel_size: int = 2
    n_layer: int = 56
    n_head: int = 16
    n_embd: int = 1792
    mlp_hidden_size: int = 15360
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"

@dataclass
class ZWM2_7BConfig(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 2048 # 32*32*2
    resolution: int = 256
    channel_size: int = 2
    n_layer: int = 48
    n_head: int = 32
    n_embd: int = 3584
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"

@dataclass
class ZWM2_170M_512px_Config(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 8192 # 64*64*2
    resolution: int = 512
    channel_size: int = 2
    n_layer: int = 24
    n_head: int = 12
    n_embd: int = 768
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"

@dataclass
class ZWM2_1B_512px_Config(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 8192 # 64*64*2
    resolution: int = 512
    channel_size: int = 2
    n_layer: int = 48
    n_head: int = 16
    n_embd: int = 1280
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"

@dataclass
class ZWM2_170M_FlexibleHW_Config(BaseConfig):
    model_class: str = "zwm.model.ZWM2"
    block_size: int = 4096 # 
    resolution: int = 256
    channel_size: int = 2
    n_layer: int = 24
    n_head: int = 12
    n_embd: int = 768
    patch_size: int = 8
    n_input_channels: int = 3
    loss_function: str = 'l2'
    dropout: float = 0.0
    bias: bool = False
    attention_mask: str = "non_causal"
    