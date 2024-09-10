import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import importlib.util
from collections import OrderedDict

if importlib.util.find_spec('flash_attn'):
    FlashMHA = importlib.import_module('flash_attn.flash_attention').FlashMHA
    
    

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head) if not use_flash_attention else FlashMHA(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if self.use_flash_attention:
            # Batch first is needed for FlashAttention. See https://github.com/HazyResearch/flash-attention/issues/84 for more information.
            return self.attn(x.transpose(1, 0))[0].transpose(1, 0)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_flash_attention: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for r in self.resblocks:
                x = checkpoint(r, x)
            return x        
        return self.resblocks(x)




class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_flash_attention: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = torch.rand(N, L - 1, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1) + torch.ones(N, L - 1, device=x.device,
                                                               dtype=int)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = torch.cat([x0, x_masked], axis=1)
        return x_masked_add

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


if __name__ == "__main__":
    # input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_flash_attention: bool = False):
    vit = VisualTransformer(
            input_resolution=224, 
            patch_size=16,
            width=768,
            layers=12,
            heads=8,
            output_dim=512,
            use_flash_attention=False
        )
    input = torch.ones(4,3, 224,224)
    out = vit(input)
    print(f'out:{out.shape}')