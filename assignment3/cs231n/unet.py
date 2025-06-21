import copy
from IPython.display import display_markdown
from einops import rearrange
from torch import einsum

from torch import nn
import torch
import torch.nn.functional as F
import math


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    """Upsample the image feature resolution a factor of 2."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    """Downsample the image feature resolution a factor of 2."""
    return nn.Conv2d(dim, default(dim_out, dim), kernel_size=2, stride=2)


class RMSNorm(nn.Module):
    """RMSNorm layer which is compute-efficient simplified variant of LayerNorm."""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal position embedding for time steps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """A conv block with feature modulation."""

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.GELU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        # Scale and shift are used to modulate the output. This is a variant
        # of feature fusion, more powerful than simply adding the feature maps.
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """A ResNet-like block with context dependent feature modulation."""

    def __init__(self, dim, dim_out, context_dim):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.context_dim = context_dim

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(context_dim, dim_out * 2))
            if exists(context_dim)
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, context=None):

        scale_shift = None
        if exists(self.mlp) and exists(context):
            context = self.mlp(context)
            context = rearrange(context, "b c -> b c 1 1")
            scale_shift = context.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        condition_dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        uncond_prob=0.2,
    ):
        super().__init__()

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)
        self.channels = channels

        # Number of channels at each layer i.e. [d1, d2, ..., dn]
        dims = [dim] + [dim * m for m in dim_mults]
        # Input and output for each U-Net block in downsampling layers
        # e.g. [(d1, d2), (d2, d3), ..., (dn-1, dn)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # Input and output for each U-Net block in upsampling layers
        # e.g. [(dn, dn-1), (dn-1, dn-2), ..., (d2, d1)]
        in_out_ups = [(b, a) for a, b in reversed(in_out)]

        # Encoding timestep as context
        context_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Encoding condition (i.e. text embedding) as context
        self.condition_dim = condition_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
        )

        # Probability of dropping the condition during training
        self.uncond_prob = uncond_prob

        # UNet downsampling and upsampling blocks.
        # self.downs is a ModuleList of ModuleLists.
        self.downs = nn.ModuleList([])
        # self.ups is a ModuleList of ModuleLists.
        self.ups = nn.ModuleList([])

        ####################################################################
        # Downsampling blocks
        ####################################################################
        for ind, (dim_in, dim_out) in enumerate(in_out):
            down_block = None
            ##################################################################
            # TODO: Create one UNet downsampling layer `down_block` as a ModuleList.
            # It should be a ModuleList of 3 blocks [ResnetBlock, ResnetBlock, Downsample].
            # Each ResnetBlock operates on dim_in channels and outputs dim_in channels.
            # Make sure to pass the context_dim to each ResnetBlock.
            # The Downsample block operates on dim_in channels and outputs dim_out channels.
            # Make sure to exactly follow this structure of ModuleList in order to
            # load a pretrained checkpoint.
            ##################################################################

            down_block = nn.ModuleList([
                ResnetBlock(dim_in, dim_in, context_dim),
                ResnetBlock(dim_in, dim_in, context_dim),
                Downsample(dim_in, dim_out),
            ])

            ##################################################################
            self.downs.append(down_block)

        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, context_dim=context_dim)

        ####################################################################
        # Upsampling blocks
        ####################################################################
        # Create upsampling blocks by exactly mirroring the downsampling blocks.
        # self.ups will also be a ModuleList of ModuleLists.
        # Each BlockList will contain 3 blocks [Upsample, ResnetBlock, ResnetBlock].
        for ind, (dim_in, dim_out) in enumerate(in_out_ups):
            up_block = None
            ##################################################################
            # TODO: Create one UNet upsampling layer as a ModuleList.
            # It should be a ModuleList of 3 blocks [Upsample, ResnetBlock, ResnetBlock].
            # This will mirror the corresponding downsampling block.
            # Don't forget to account for the skip connections by having 2 x dim_out
            # channels at the input of both ResnetBlocks.
            ##################################################################

            up_block = nn.ModuleList([
                Upsample(dim_in, dim_out),
                ResnetBlock(2 * dim_out, dim_out, context_dim),
                ResnetBlock(2 * dim_out, dim_out, context_dim),
            ])

            self.ups.append(up_block)
            ##################################################################

        # Final convolution to map to the output channels
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def cfg_forward(self, x, time, model_kwargs={}):
        """Classifier-free guidance forward pass. model_kwargs should contain `cfg_scale`."""

        cfg_scale = model_kwargs.pop("cfg_scale")
        print("Classifier-free guidance scale:", cfg_scale)
        model_kwargs = copy.deepcopy(model_kwargs)

        ##################################################################
        # TODO: Apply classifier-free guidance using Eq. (6) from
        # https://arxiv.org/pdf/2207.12598 i.e.
        # x = (scale + 1) * eps(x_t, cond) - scale * eps(x_t, empty)
        #
        # You will have to call self.forward two times.
        # For unconditional sampling, pass None in`text_emb`.
        ##################################################################

        x_cond = self.forward(x, time, model_kwargs)
        x_uncond = self.forward(x, time, {**model_kwargs, "text_emb": None})
        x = (1 + cfg_scale) * x_cond - cfg_scale * x_uncond

        ##################################################################

        return x

    def forward(self, x, time, model_kwargs={}):
        """Forward pass through the U-Net.
        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            time: Tensor of time steps of shape (batch_size,).
            model_kwargs: A dictionary of additional model inputs including
                "text_emb" (text embedding) of shape (batch_size, condition_dim).

        Returns:
            x: Output tensor of shape (batch_size, channels, height, width).
        """

        if "cfg_scale" in model_kwargs:
            return self.cfg_forward(x, time, model_kwargs)

        # Embed time step
        context = self.time_mlp(time)

        # Embed condition and add to context
        cond_emb = model_kwargs["text_emb"]
        if cond_emb is None:
            cond_emb = torch.zeros(x.shape[0], self.condition_dim, device=x.device)
        if self.training:
            # Randomly drop condition
            mask = (torch.rand(cond_emb.shape[0]) > self.uncond_prob).float()
            mask = mask[:, None].to(cond_emb.device)  # B x 1
            cond_emb = cond_emb * mask
        context = context + self.condition_mlp(cond_emb)

        # Initial convolution
        x = self.init_conv(x)

        ##################################################################
        # TODO: Process `x` through the U-Net conditioned on the context.
        #
        # 1. Downsampling:
        #    - Process `x` through each downsampling block with context.
        #    - After each ResNet block, save the output (feature maps) in a list or dict
        #      for use as skip connections in the upsampling path.
        #    - Make sure to pass the context to each ResNet block.
        #
        # 2. Middle:
        #    - Process `x` through the middle blocks with context.
        #
        # 3. Upsampling:
        #    - Process `x` through each upsampling block with context.
        #    - Before each ResNet block, concatenate the input with the corresponding
        #      skip connection from the downsampling path.
        #    - Make sure to pass the context to each ResNet block.
        ##################################################################

        # Init downs
        x_downs = []

        for down_block in self.downs:
            # Save the output of the :-1 blocks for residual connections
            x_downs.append([x := b(x, context) for b in down_block[:-1]])
            x = down_block[-1](x)

        # Apply the middle blocks in sequence with given context
        x = self.mid_block2(self.mid_block1(x, context), context)

        for up_block, xs_down in zip(self.ups, reversed(x_downs)):
            # Upsample, make residual input pairs, apply res-blocks
            x, pairs = up_block[0](x), zip(up_block[1:], xs_down[::-1])
            [x := b(torch.cat([x, x0], dim=1), context) for b, x0 in pairs]

        ##################################################################

        # Final block
        x = self.final_conv(x)

        return x
