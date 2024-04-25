# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import warnings

from classification.models.vmamba import Backbone_VSSM, VSSM
from davit.timm import create_model
from utils import logging
import torch
from einops import rearrange
from timm.models.vision_transformer import trunc_normal_, default_cfgs, _cfg, HybridEmbed, partial, OrderedDict
from torch.hub import load_state_dict_from_url

from models.transformer_base.encoder.transformer_block import PatchEmbed, Block

default_cfgs['vit_deit_base_mae_patch16_224'] = \
    _cfg('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth')



import torch
from mamba_ssm import Mamba
class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()

        if torch.distributed.get_rank() == 0:
            print('Base Encoder: VIT')

        self.model=create_model("DaViT_base",pretrained=True,checkpoint_path="/root/autodl-tmp/R3DSWIN++/davit/model_best.pth.tar")





    def forward(self, x):
        batch_size = x.shape[0]
        view = x.shape[1]
        x = rearrange(x, 'b v c h w -> (b v) c h w').contiguous()  # [B, V*P, D]

        x = self.model(x)

        x = rearrange(x, 'b (v h w) d -> b v h w d', b=batch_size, v=view, h=7, w=7).contiguous()  # [B, V*P, D]8 147 768
        x = x.mean(dim=1)

        return x


vit_cfg = dict(
    vit_small_patch16_224=dict(patch_size=16, embed_dim=768, depth=8, num_heads=8,
                               mlp_ratio=3., qkv_bias=False, norm_layer=torch.nn.LayerNorm),
    vit_base_patch16_224=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    vit_base_patch32_224=dict(patch_size=32, embed_dim=768, depth=12, num_heads=12),
    vit_base_patch16_384=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    vit_base_patch32_384=dict(patch_size=32, embed_dim=768, depth=12, num_heads=12),
    vit_large_patch16_224=dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    vit_large_patch32_224=dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16),
    vit_large_patch16_384=dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
    vit_large_patch32_384=dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16),
    vit_base_patch16_224_in21k=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=768),
    vit_base_patch32_224_in21k=dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, representation_size=768),
    vit_large_patch16_224_in21k=dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, representation_size=1024),
    vit_large_patch32_224_in21k=dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024),
    vit_huge_patch14_224_in21k=dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280),
    vit_deit_tiny_patch16_224=dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    vit_deit_small_patch16_224=dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    vit_deit_base_patch16_224=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    vit_deit_base_patch16_384=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    vit_deit_tiny_distilled_patch16_224=dict(patch_size=16, embed_dim=192, depth=12, num_heads=3),
    vit_deit_small_distilled_patch16_224=dict(patch_size=16, embed_dim=384, depth=12, num_heads=6),
    vit_deit_base_distilled_patch16_224=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    vit_deit_base_distilled_patch16_384=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    vit_deit_base_mae_patch16_224=dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
)


def load_state_dict_partially(model, state_dict):
    own_state = model.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            if 'attn.qkv.weight' in name:
                load_param(param[:768, :], own_state, name.replace('qkv', 'to_q'))
                load_param(param[768:768 * 2, :], own_state, name.replace('qkv', 'to_k'))
                load_param(param[768 * 2:, :], own_state, name.replace('qkv', 'to_v'))
            elif 'attn.qkv.bias' in name:
                load_param(param[:768], own_state, name.replace('qkv', 'to_q'))
                load_param(param[768:768 * 2], own_state, name.replace('qkv', 'to_k'))
                load_param(param[768 * 2:], own_state, name.replace('qkv', 'to_v'))
            else:
                logging.info(f'Ignored parameter "{name}" on loading')
                continue
        else:
            load_param(param, own_state, name)


def load_param(param, own_state, name):
    if isinstance(param, torch.nn.Parameter):
        # backwards compatibility for serialized parameters
        param = param.data
    try:
        own_state[name].copy_(param)
    except RuntimeError:
        print(f'Ignored parameter "{name}" on loading')


class VisionTransformer(torch.nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            representation_size=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            hybrid_backbone=None,
            norm_layer=None,
            use_cls_token=True,
            patch_group=7,
            lga_layer=[5, 7, 9, 11]):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(torch.nn.LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_cls_token = use_cls_token
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule ???
        self.blocks = torch.nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  attn_drop=attn_drop_rate, proj_drop=drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, patch_group=patch_group) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.lga_layer = lga_layer

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = torch.nn.Sequential(OrderedDict([
                ('fc', torch.nn.Linear(embed_dim, representation_size)),
                ('act', torch.nn.Tanh())
            ]))
        else:
            self.pre_logits = torch.nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def prepare(self, x):
        batch_size, view_num = x.shape[0:2]
        x = rearrange(x, 'b v c h w -> (b v) c h w')  # [B*V, C, H, W]
        x = self.patch_embed(x)  # [B*V, P, D]
        # Remove CLS token
        x = x + self.pos_embed[:, 1:, :]  # [B*V, P, D]
        x = self.pos_drop(x)
        return x, batch_size, view_num

    def forward(self, x):
        # x [B, V, C, H, W]
        x, batch_size, view_num = self.prepare(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def load_pretrained(self, model_name):
        state_dict = load_state_dict_from_url(default_cfgs[model_name]['url'], progress=True)
        load_state_dict_partially(self, state_dict)





class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.dist_token = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        # self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        if torch.distributed.get_rank() == 0:
            print('Encoder: Long-Range Grouping Transformer (LRGT)')

        # trunc_normal_(self.dist_token, std=.02)
        # trunc_normal_(self.pos_embed, std=.02)



    def prepare(self, x):
        # print(f'\n x begin in DVIT{x.shape}')
        batch_size, view_num = x.shape[0:2]
        x = rearrange(x, 'b v c h w -> (b v) c h w')  # [B*V, C, H, W]
        # print(f'\n x after rearrange :{x.shape}')
        x = self.patch_embed(x)  # [B*V, P-2, D]
        # print(f'\n x after patch_embed:{x.shape}')
        # Remove CLS token
        x = x + self.pos_embed[:, 2:, :]  # [B*V, P, D]
        x = self.pos_drop(x)
        #  print(f'\n prepare return x :{x.shape}')
        return x, batch_size, view_num

    def forward(self, x):
        # x [B, V, C, H, W]
        print(f'\n x begin in encoder:{x.shape}')  # x begin in encoder:torch.Size([16, 1, 3, 224, 224])
        _, batch_size, view_num = self.prepare(x)
        # print(f'\n x  after prepare :{x.shape}')
        # x=rearrange('b (hw) d->b d h w')
        # print(f'\n x  after rearge :{x.shape}')
        x = x.squeeze(dim=1)
        print(f'\n x after squeeze:{x.shape}')
        x = self.model(x)
        print(f'\n x after model:{x.shape}')
        # x  after prepare :torch.Size([16, 196, 768])
        # layer = 0
        # for blk in self.blocks:
        #     if layer in self.lga_layer:# lga_layer=[5, 7, 9, 11]
        #         print(f'\n lga layer:{layer}')
        #         #x begin in convit:torch.Size([16, 3, 224, 224])

        #         # x in convit after embedding:torch.Size([16, 786, 14, 14])
        #         # x in transformer:torch.Size([16, 786, 14, 14])
        #         #
        #         x = blk(x, view_num=view_num, lga=True)
        #     else:
        #         x = blk(x)
        #     layer += 1
        # x = self.norm(x)
        print(f'\n encoder  return x :{x.shape} ')
        return x

    def load_pretrained(self, model_name):
        state_dict = load_state_dict_from_url(default_cfgs[model_name]['url'], progress=True)['model']
        load_state_dict_partially(self, state_dict)
