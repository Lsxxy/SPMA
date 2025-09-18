""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
from dataclasses import dataclass
import logging
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch import nn
from torch.utils.checkpoint import checkpoint
from collections import defaultdict

from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, QuickGELU, Attention, VisionTransformer, TextTransformer
from .utils import to_2tuple
from GrootVL.GrootV.classification.models.tree_scanning import Tree_SSM
from GrootVL.GrootL.transformers1.src.transformers.models.mamba.modeling_mamba import MambaBlock,MambaMixer
from GrootVL.GrootL.tree_scanning import tree_scanning_algorithm
from einops import rearrange
from hydramain.hydra import Hydra


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        MMadapter_img: Optional[nn.ModuleList] = None,
        MMadapter_aux: Optional[nn.ModuleList] = None,
        modalemb: Optional[nn.Module] = None,
        adapter_prompt: Optional[bool] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size
        )
        act_layer = nn.GELU  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else nn.LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mmadapter=MMadapter_img,
            mmadapter_aux=MMadapter_aux,
            adapter_prompt =adapter_prompt,
            modalemb=None
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        MMadapter_text: Optional[nn.ModuleList] = None,
        MMadapter_aux: Optional[nn.ModuleList] = None,
        adapter_prompt:Optional[bool] = None,
        modalemb: Optional[nn.Module] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else nn.LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        mmadapter=MMadapter_text,
        adapter_prompt = adapter_prompt,
        mmadapter_aux=MMadapter_aux,
        modalemb=None
    )

    return text






def pre_data(x):
    #batch, seqlen, dim = x.shape
    h=7
    hidden_states_image_patches = rearrange(x,"b (h w) d -> b h w d",h=h) # (batch_size,h,w，embedding_dim)，取出图像patch信息，这些信息进行后续计算
    return hidden_states_image_patches




#######Treemamba ImageAdapter#####################################################
class IMGMMadapter(nn.Module):
    """
    A module that implements a multimodal adapter with shared components and multi-head attention.
    
    Attributes:
        img_proj_down (nn.Linear): Linear layer that projects the input to a lower dimension.
        img_proj_up (nn.Linear): Linear layer that projects back to the original dimension.
        BiShareAdapterxx (BiShareAdapter or None): Optional shared BiShareAdapter instance.
        multihead_attention (nn.MultiheadAttention): Multi-head attention layer.
        gate1 (nn.Parameter): A learnable gate parameter for blending attention output with input.
    """
    def __init__(self, share_adapter, hidden_size, layer_id=0, share_prompt=None):
        """
        Inits MMadapter with a shared adapter, hidden size, and layer ID.
        
        Args:
            share_adapter (BiShareAdapter or None): The shared BiShareAdapter instance, if any.
            hidden_size (int): The size of the hidden layer.
            layer_id (int, optional): The layer ID, defaults to 0.
        """
        super(IMGMMadapter, self).__init__()
        self.img_proj_down = nn.Linear(hidden_size, 128)
        self.img_proj_up = nn.Linear(128, hidden_size)
        self.mode = 'visual'
        self.BiShareAdapterxx = share_adapter
        self.share_prompt = share_prompt
        self.TreeSSM = Tree_SSM(
                d_model=128, 
                d_state=1, 
                ssm_ratio=2,
                ssm_rank_ratio=2,
                dt_rank='auto',
                act_layer=nn.SiLU,
                d_conv=3,
                conv_bias=False,
                dropout=0.0,
            )
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        """Initializes weights with zeros."""
        nn.init.kaiming_uniform_(self.img_proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.img_proj_down.bias)
        nn.init.zeros_(self.img_proj_up.weight)
        nn.init.zeros_(self.img_proj_up.bias)
        

    def forward(self, x, layer_prompt:Optional[torch.Tensor] = None):
        """
        Forward pass of the MMadapter.
        
        Args:
            x (Tensor): The input tensor.
            
        Returns:
            Tensor: The output tensor after the adapter processing.
        """
        x_init = x
        x = self.img_proj_down(x)#(L,B,128)
        if self.share_prompt == True and layer_prompt is not None:
            x = torch.cat([x,layer_prompt],dim=0) 
        x = x.permute(1,0,2) #(B,L=n**2+1,128)
        x = F.gelu(x)
        xmid = x
        img_patches = pre_data(x)
        x = self.TreeSSM(img_patches)
        x = rearrange(self.TreeSSM(x),"b h w d -> b (h w) d")
        
        alpha = torch.sigmoid(self.gate1)
        x = alpha * xmid + (1 - alpha) * x
        x = x+xmid
        if self.share_prompt == True and layer_prompt is not None:
            x = x[:,:-6,:] 
        x = self.img_proj_up(x) #(B,L,D)
        x = x.permute(1,0,2)  #(L,B,D)
        x = x_init + x

        return x

#######Hydra TextAdapter#################################################################
class TextMMadapter(nn.Module):

    def __init__(self, share_adapter, hidden_size, layer_id=0, share_prompt=None):

        super(TextMMadapter, self).__init__()
        self.img_proj_down = nn.Linear(hidden_size, 128)
        self.img_proj_up = nn.Linear(128, hidden_size)
        self.BiShareAdapterxx = share_adapter
        self.share_prompt = share_prompt
        self.mode = 'text'    
        self.hydra = Hydra(
                d_model=128, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=7,    # Local non-causal convolution width
                expand=2,    # Block expansion factor
                use_mem_eff_path=False,    
            )
        self.gate1 = nn.Parameter(torch.tensor(0.6), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        """Initializes weights with zeros."""
        nn.init.kaiming_uniform_(self.img_proj_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.img_proj_down.bias)
        nn.init.zeros_(self.img_proj_up.weight)
        nn.init.zeros_(self.img_proj_up.bias)

    def forward(self, x, layer_prompt:Optional[torch.Tensor] = None):
        """
        Forward pass of the MMadapter.
        
        Args:
            x (Tensor): The input tensor.
            
        Returns:
            Tensor: The output tensor after the adapter processing.
        """
        #（L,B,D）
        x_init = x
        x = self.img_proj_down(x) #(L,B,128)
        if self.share_prompt == True and layer_prompt is not None:
            x = torch.cat([layer_prompt,x],dim=0) 
        x = x.permute(1,0,2) #(B,L,128)
        x = F.gelu(x)
        xmid = x
        x = self.hydra(x)
        x = self.hydra(x)
        alpha = torch.sigmoid(self.gate1)
        x = alpha * xmid + (1 - alpha) * x
        x = xmid + x
        if self.share_prompt == True and layer_prompt is not None:
            x = x[:,6:,:] 
        x = self.img_proj_up(x)
        x = x.permute(1,0,2) #(L,B,D)
        x = x_init + x

        return x


    
  
MMadapter_img = nn.ModuleList([
    IMGMMadapter(None,hidden_size=768,layer_id=layer_id,share_prompt=True)
    for layer_id in range(12)
])
MMadapter_text = nn.ModuleList([
    TextMMadapter(None,hidden_size=512,layer_id=layer_id,share_prompt=True)
    for layer_id in range(12)
]) 

class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_replicas, stack=False, is_array=True):
        super(ProjectLayer, self).__init__()

        self.head = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_replicas)])
        self.num_replicas = num_replicas
        self.stack = stack
        self.is_array = is_array

    def forward(self, tokens):
        out_tokens = []
        for i in range(self.num_replicas):
            if self.is_array:
                temp = self.head[i](tokens[i][:, 1:, :]) # for ViT, we exclude the class token and only extract patch tokens here.
            else:
                temp = self.head[i](tokens)

            out_tokens.append(temp)

        if self.stack:
            out_tokens = torch.stack(out_tokens, dim=1)

        return out_tokens


class PrePromptCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype,MMadapter_img=None,MMadapter_aux=None,modalemb=None)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype,MMadapter_text=None,MMadapter_aux=None,modalemb=None)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.align_before = False

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        if self.align_before:
            features,vis_fea = self.visual(image)
        else:
            features = self.visual(image)
        if normalize:
            features = F.normalize(features, dim=-1)
            if self.align_before:
                vis_fea = [F.normalize(fea, dim=-1) for fea in vis_fea] 
                return features, vis_fea
        return features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.align_before:
            x,intermediates_text = self.transformer(x, attn_mask=self.attn_mask)
        else:
            x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        if self.align_before:
            for i in range(len(intermediates_text)):
                intermediates_text[i] = intermediates_text[i][torch.arange(intermediates_text[i].shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if normalize:
            x = F.normalize(x, dim=-1)
            if self.align_before:
                for i in range(len(intermediates_text)):
                    intermediates_text[i] = F.normalize(intermediates_text[i], dim=-1)
                return x, intermediates_text
        return x
    

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features,text_features




class CLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        #dynamic_prompt ProjectLayer
        self.visual_project_layer_for_dynamic_prompt = ProjectLayer(input_dim=512,output_dim=128,num_replicas=1,stack=True,is_array=False)
        self.text_project_layer_for_dynamic_prompt = ProjectLayer(input_dim=512,output_dim=128,num_replicas=1,stack=True,is_array=False)

        #static_prompt
        self.share_static_prompts = nn.ParameterList([nn.Parameter(torch.empty(5, 128))
                                               for _ in range(12)])
        for single_para in self.share_static_prompts:
            nn.init.normal_(single_para, std=0.02)

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype,MMadapter_img=MMadapter_img,adapter_prompt=True,MMadapter_aux=None,modalemb=None)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype,MMadapter_text=MMadapter_text,adapter_prompt=True,MMadapter_aux=None,modalemb=None)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.align_before = False

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, img_preprompt:torch.tensor = None,normalize: bool = False):
        visual_final_prompt = []
        if self.align_before:
            features,vis_fea = self.visual(image)
        else:
            #visual部分的prompt
            if img_preprompt == None :
                img_preprompt = self.viusal_for_dynamic_prompt(image)
            image_dynamic_prompt = self.visual_project_layer_for_dynamic_prompt(img_preprompt)
            for index in range(12):
                static_prompts = self.share_static_prompts[index].unsqueeze(0).expand(img_preprompt.shape[0], -1, -1)
                visual_context = torch.cat((image_dynamic_prompt,static_prompts),dim=1)
                visual_final_prompt.append(visual_context)
            viusal_final_prompt = torch.stack(visual_final_prompt, dim=1)
            features = self.visual(image,viusal_final_prompt)

        if normalize:
            features = F.normalize(features, dim=-1)
            if self.align_before:
                vis_fea = [F.normalize(fea, dim=-1) for fea in vis_fea] 
                return features, vis_fea
        return features

    def encode_text(self, text, text_preprompt:torch.tensor = None,normalize: bool = False):
        text_final_prompt = []
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.align_before:
            x,intermediates_text = self.transformer(x, attn_mask=self.attn_mask)
        else:
            if text_preprompt == None:
                text_preprompt = self.encode_prompt_embed(text)
            text_dynamic_prompt = self.text_project_layer_for_dynamic_prompt(text_preprompt)
            for index in range(12):
                static_prompts = self.share_static_prompts[index].unsqueeze(0).expand(text_preprompt.shape[0], -1, -1)
                text_context = torch.cat((text_dynamic_prompt,static_prompts),dim=1)
                text_final_prompt.append(text_context)
            text_final_prompt = torch.stack(text_final_prompt, dim=1)
            x = self.transformer(x, attn_mask=self.attn_mask,final_prompt=text_final_prompt)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        if self.align_before:
            for i in range(len(intermediates_text)):
                intermediates_text[i] = intermediates_text[i][torch.arange(intermediates_text[i].shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if normalize:
            x = F.normalize(x, dim=-1)
            if self.align_before:
                for i in range(len(intermediates_text)):
                    intermediates_text[i] = F.normalize(intermediates_text[i], dim=-1)
                return x, intermediates_text
        return x
    
    def encode_prompt_embed(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.align_before:
            x,intermediates_text = self.transformer(x, attn_mask=self.attn_mask)
        else:
            x = self.text_for_dynamic_prompt(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        if self.align_before:
            for i in range(len(intermediates_text)):
                intermediates_text[i] = intermediates_text[i][torch.arange(intermediates_text[i].shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if normalize:
            x = F.normalize(x, dim=-1)
            if self.align_before:
                for i in range(len(intermediates_text)):
                    intermediates_text[i] = F.normalize(intermediates_text[i], dim=-1)
                return x, intermediates_text
        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict,strict=False)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed
    
    
    
    

