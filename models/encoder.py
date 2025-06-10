from typing import Optional
import timm
import torch
import torch.nn as nn
from timm.layers import (
    resample_patch_embed,
    resample_abs_pos_embed,
    resample_abs_pos_embed_nhwc,
)
from timm.models._manipulate import checkpoint_seq
from torch.nn.functional import interpolate
from open_clip import create_model_from_pretrained, get_tokenizer
import sys;sys.path.append('/home/manugaur/mllm_inversion')
from src.models import FlamingoCrossAttn
from .lora import lora_siglip
import os

class Encoder(nn.Module):
    
    def load_checkpoint(self, model, ckpt_dir, ckpt_name):
        pretrained_wts = torch.load(os.path.join(ckpt_dir, ckpt_name))['model_state_dict']
        num_weights = len(pretrained_wts)
        print(len(pretrained_wts))
        
        for k,v in pretrained_wts.copy().items():
            if "visual_encoder.siglip.visual" in k:
                new_k = k.replace("visual_encoder.siglip.visual", "visual_encoder")
                del pretrained_wts[k]
                pretrained_wts[new_k] = v        
        print(len(pretrained_wts))
        
        weights_loaded = set()
        state_dict = {}
        c = 0
        for k,v in model.state_dict().items():
            if k in pretrained_wts: 
                state_dict[k] = pretrained_wts[k]
                weights_loaded.add(k)
                c+=1
            else:
                state_dict[k] = v
        print(f"params that can't be loaded into the model :\n {set(list(pretrained_wts.keys())) - weights_loaded}")

        # model.load_state_dict(state_dict)
        print(f"| LOADED {c}/{num_weights} WEIGHTS")

    def __init__(
        self,
        text_conditioning,
        encoder_name,
        img_size: tuple[int, int],
        original_res,
        ckpt_name,
        sub_norm,
        patch_size,
        pretrained,
    ):
        super().__init__()
        self.text_conditioning = text_conditioning
        if 'so400m' in encoder_name.lower():
            patch_size = 14
            if text_conditioning:
                self.encoder = FlamingoCrossAttn(
                        visual_encoder="siglip",
                        text_encoder ="roberta",
                        img_res = original_res,
                        patch_size = patch_size,
                        cross_attn_layers =[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
                        cross_attn_ffn_mult =2,
                        )
                lora_siglip(
                self.encoder,
                rank = 8,
                last_n_blocks=6
                )
                self.load_checkpoint(
                    self.encoder,
                    ckpt_dir = "/storage/users/manugaur/mllm_inversion/checkpoints",
                    ckpt_name = ckpt_name,
                    )
                import ipdb;ipdb.set_trace()

                if (img_size[0] % patch_size) != 0:
                    self.encoder.visual_encoder.trunk.patch_embed.dynamic_img_pad = True
                # self.model = self.encoder
                # self.encoder = self.encoder.visual_encoder.trunk

            else:
                encoder_name = f"hf-hub:timm/{encoder_name}"
                self.encoder, self.preprocess = create_model_from_pretrained(encoder_name)
                self.encoder = self.encoder.visual.trunk
                self.encoder.patch_embed.dynamic_img_pad = True
            
            norm_mean = norm_std = (0.5, 0.5, 0.5)
            self.encoder.embed_dim = 1152

        else:
            model_kwargs = {
                "model_name": encoder_name,
                "pretrained": pretrained,
                "num_classes": 0,
            }

            self.encoder = timm.create_model(**model_kwargs)
            norm_mean = self.encoder.default_cfg["mean"]
            norm_std = self.encoder.default_cfg["std"]
        
        pixel_mean = torch.tensor(norm_mean).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(norm_std).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self.grid_size = tuple(round(size / patch_size) for size in img_size) #should be (37,37) when patch_size=14

        self.embed_dim = (
            self.encoder.embed_dim
            if hasattr(self.encoder, "embed_dim")
            else self.encoder.num_features
        )
        if sub_norm:
            for block in self.encoder.blocks:
                new_mlp = type(block.mlp)(
                    in_features=block.mlp.fc1.in_features,
                    hidden_features=block.mlp.fc1.out_features,
                    act_layer=type(block.mlp.act),
                    drop=block.mlp.drop1.p,
                    norm_layer=nn.LayerNorm,
                )
                new_mlp.load_state_dict(block.mlp.state_dict(), strict=False)
                block.mlp = new_mlp
                block.attn.proj = nn.Sequential(
                    nn.LayerNorm(block.attn.proj.in_features), block.attn.proj
                )

        if hasattr(self.encoder, "neck"):
            self.encoder.neck = nn.Identity()
        if ckpt_path:
            self.encoder.load_state_dict(torch.load(ckpt_path))

        if hasattr(self.encoder, "rope"):
            self.encoder.rope = timm.create_model(
                img_size=img_size, patch_size=patch_size, **model_kwargs
            ).rope

        if hasattr(self.encoder, "blocks"):
            for block in self.encoder.blocks:
                old_window_size = None
                if hasattr(block, "window_size"):
                    old_window_size = block.window_size
                    window_ratio = (
                        old_window_size / self.encoder.patch_embed.grid_size[0]
                    )
                    new_window_size = window_ratio * (img_size[0] / patch_size)

                    if new_window_size != round(new_window_size):
                        raise ValueError("invalid window size")

                    block.window_size = int(new_window_size)

                if hasattr(block.attn, "rel_pos_h"):
                    block.attn.rel_pos_h = self.interpolate_rel_pos(
                        block.attn.rel_pos_h,
                        img_size[0] / patch_size,
                        self.encoder.patch_embed.grid_size[0],
                        block.window_size,
                        old_window_size,
                    )

                if hasattr(block.attn, "rel_pos_w"):
                    block.attn.rel_pos_w = self.interpolate_rel_pos(
                        block.attn.rel_pos_w,
                        img_size[1] / patch_size,
                        self.encoder.patch_embed.grid_size[1],
                        block.window_size,
                        old_window_size,
                    )

        if hasattr(self.encoder, "patch_embed"):
            if (
                self.encoder.patch_embed.grid_size[0]
                != self.encoder.patch_embed.grid_size[1]
                or self.encoder.patch_embed.patch_size[0]
                != self.encoder.patch_embed.patch_size[1]
            ):
                raise ValueError("pretrained grid and patch size must be square")
            self.encoder.patch_embed.patch_size = (patch_size, patch_size) #change patch_size
            self.encoder.patch_embed.proj.kernel_size = (patch_size, patch_size)
            self.encoder.patch_embed.proj.stride = (patch_size, patch_size)
            self.encoder.patch_embed.proj.weight = nn.Parameter(
                resample_patch_embed(
                    self.encoder.patch_embed.proj.weight,
                    [patch_size, patch_size],
                )
            )#interpolate patch_embed.weight

            self.encoder.patch_embed.grid_size = self.grid_size
            self.encoder.patch_embed.num_patches = self.grid_size[0] * self.grid_size[1]
            self.encoder.patch_embed.img_size = img_size
        
        if hasattr(self.encoder, "pos_embed"):
            if self.encoder.pos_embed.dim() == 4:
                pos_embed = resample_abs_pos_embed_nhwc(
                    self.encoder.pos_embed, [max(self.grid_size), max(self.grid_size)]
                )[:, : self.grid_size[0], : self.grid_size[1], :]
            else:
                num_prefix_tokens = (
                    0
                    if getattr(self.encoder, "no_embed_class", False)
                    else self.encoder.num_prefix_tokens
                )
                pos_embed = resample_abs_pos_embed(
                    self.encoder.pos_embed,
                    [
                        max(self.grid_size),
                        max(self.grid_size),
                    ],
                    num_prefix_tokens=num_prefix_tokens,
                )
                prefix_pos_embed = pos_embed[:, :num_prefix_tokens, :]
                pos_embed = pos_embed[:, num_prefix_tokens:, :]
                pos_embed = pos_embed.reshape(
                    1, max(self.grid_size), max(self.grid_size), -1
                )[:, : self.grid_size[0], : self.grid_size[1], :]
                pos_embed = torch.cat(
                    [prefix_pos_embed, pos_embed.flatten(1, 2)], dim=1
                )

            self.encoder.pos_embed = nn.Parameter(pos_embed)

    @staticmethod
    def interpolate_rel_pos(
        rel_pos, grid_size, old_grid_size, window_size=None, old_window_size=None
    ):
        block_size = (rel_pos.shape[0] + 1) / 2

        if block_size == old_grid_size:
            max_rel_dist = grid_size * 2 + 1
        elif block_size == old_window_size:
            if window_size is None:
                raise ValueError("window_size must be specified for non-global blocks")

            max_rel_dist = window_size * 2 + 1
        else:
            raise ValueError("invalid block size")

        max_rel_dist = int(max_rel_dist)

        rel_pos = rel_pos.reshape(1, rel_pos.shape[0], -1)
        rel_pos = rel_pos.permute(0, 2, 1)
        rel_pos = interpolate(rel_pos, size=max_rel_dist, mode="linear")
        rel_pos = rel_pos.reshape(-1, max_rel_dist).permute(1, 0)

        return nn.Parameter(rel_pos)

    def forward(self, x: torch.Tensor, text_cond=None):
        x = (x - self.pixel_mean) / self.pixel_std
        
        if self.text_conditioning and (text_cond is not None):
            x = self.encoder(x, text_cond[0], text_cond[1], get_feats=True)
        else:
            x = self.encoder.forward_features(x)
            if x.dim() == 4:
                x = x.flatten(2).transpose(1, 2)
            else:
                x = x[:, self.encoder.num_prefix_tokens :]

        return x