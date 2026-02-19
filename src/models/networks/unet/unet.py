"""
Unified UNet model consolidation.
Replaces unet.py, unet_bb.py, unet_class_free.py, unet_avg_vel.py.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..shared.fp16_util import convert_module_to_f16, convert_module_to_f32

from ..shared.unet_blocks import TimestepEmbedSequential, ResBlock, Upsample, Downsample
from ..shared.attention import AttentionBlock
from ..shared.layers import (
    conv_nd,
    linear,
    normalization,
    zero_module,
    SiLU
)
from ..shared.embeddings import timestep_embedding


NUM_CLASSES = 1000


class UNetModel(nn.Module):
    """The full UNet model with attention and timestep embedding.

    Unified implementation supporting:
    - Standard Time+Class conditioning
    - Unconditional Backbone (unet_bb) via use_time_emb=False
    - Classifier-Free Guidance Tweaks (unet_class_free) via use_delta_y_emb=True
    - Future Time Embedding (unet_avg_vel) via use_future_time_emb=True
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # New unification flags
        use_time_emb=True,     # set False to mimic unet_bb
        use_future_time_emb=False, # set True for unet_avg_vel
        use_delta_y_emb=False,     # set True for unet_class_free
        delta_y_prob=0.1,          # for unet_class_free
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # Unification Flags
        self.use_time_emb = use_time_emb
        self.use_future_time_emb = use_future_time_emb
        self.use_delta_y_emb = use_delta_y_emb
        self.delta_y_prob = delta_y_prob

        time_embed_dim = model_channels * 4

        # 1. Timestep Embedding
        if self.use_time_emb:
            self.time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.time_embed = None

        # 2. Future Timestep Embedding (unet_avg_vel)
        if self.use_future_time_emb:
            self.future_time_embed = nn.Sequential(
                linear(model_channels, time_embed_dim),
                SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            self.future_time_embed = None

        # 3. Label Embedding
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # 4. Delta Y Embedding (unet_class_free)
        if self.use_delta_y_emb and self.num_classes is not None:
            self.delta_y_embed = nn.Embedding(num_classes, time_embed_dim)
            self.register_buffer("delta_y_dropout", th.tensor(delta_y_prob))
        else:
            self.delta_y_embed = None


        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        have_time_emb=self.use_time_emb
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            have_time_emb=self.use_time_emb
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                have_time_emb=self.use_time_emb
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                have_time_emb=self.use_time_emb
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        have_time_emb=self.use_time_emb
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            have_time_emb=self.use_time_emb
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

        # Inference mode flag for delta_y logic
        self.infer_mode = ~self.training

    def convert_to_fp16(self):
        """Convert the torso of the model to float16."""
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the torso of the model to float32."""
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def set_infer_mode(self, mode=True):
        self.infer_mode = mode

    def forward(self, t, x, y=None, r=None, *args, **kwargs):
        """Apply the model to an input batch.

        :param t: a 1-D batch of timesteps.
        :param x: an [N x C x ...] Tensor of inputs.
        :param y: an [N] Tensor of labels, if class-conditional.
        :param r: a 1-D batch of future timesteps (for unet_avg_vel).
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        timesteps = t

        # Handle time embedding
        if t is not None and self.use_time_emb:
            while timesteps.dim() > 1:
                timesteps = timesteps[:, 0]
            if timesteps.dim() == 0:
                timesteps = timesteps.repeat(x.shape[0])
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        else:
            emb = None

        # Handle future time embedding (r)
        if self.use_future_time_emb and r is not None:
             # Logic from unet_avg_vel
             future_timesteps = r
             while future_timesteps.dim() > 1:
                future_timesteps = future_timesteps[:, 0]
             if future_timesteps.dim() == 0:
                future_timesteps = future_timesteps.repeat(x.shape[0])

             future_emb = self.future_time_embed(timestep_embedding(future_timesteps, self.model_channels))
             if emb is not None:
                 emb = emb + future_emb
             else:
                 emb = future_emb

        # Handle class/delta_y embedding
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            label_emb = self.label_emb(y)

            # Back to standard UNet behavior + delta tweaks
            if emb is not None:
                emb = emb + label_emb
            else:
                emb = label_emb

            # Apply delta y if needed (on top of checks above)
            if self.use_delta_y_emb:
                 delta_y = self.delta_y_embed(y)
                 if self.infer_mode:
                     emb = emb + delta_y
                 else:
                     mask = (th.rand((emb.shape[0], 1), device=emb.device) > self.delta_y_prob).float()
                     # If mask is 1 (prob > threshold), we add delta_y. If 0 (dropped), we don't.
                     emb = emb + delta_y * mask

        h = x.type(self.dtype)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(timesteps, x, **kwargs) # Note: argument order fixed (t, x)

class UNetModelWrapper(UNetModel):
    def __init__(
        self,
        dim,
        num_res_blocks,
        num_channels=None, # Legacy arg support
        model_channels=None, # Unified arg
        out_channels=None,
        channel_mult=None,
        learn_sigma=False,
        class_cond=False,
        num_classes=NUM_CLASSES,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        **kwargs,
    ):
        """Wrapper for backward compatibility and default configuration logic.

        Handles:
        - dim tuple -> image_size, in_channels
        - channel_mult defaults based on image_size
        - string parsing for channel_mult and attention_resolutions
        - out_channels calculation based on learn_sigma
        """
        # Handle dim tuple
        image_size = dim[-1]
        in_channels = dim[0]

        # Handle model_channels / num_channels alias
        if model_channels is None:
             if num_channels is not None:
                 model_channels = num_channels
             else:
                 raise ValueError("Must specify model_channels or num_channels")

        # Handle channel_mult defaults
        if channel_mult is None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (1, 2, 2)
            elif image_size == 200:
                 channel_mult = (1, 2, 4, 4)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            if isinstance(channel_mult, str):
                channel_mult = tuple(map(int, channel_mult.split(', ')))

        # Handle attention_resolutions string parsing
        if isinstance(attention_resolutions, str):
            attention_ds = []
            for res in attention_resolutions.split(","):
                attention_ds.append(dim[1] // int(res))
            attention_resolutions = tuple(attention_ds)

        # Handle out_channels
        if out_channels is None:
             out_channels = in_channels if not learn_sigma else in_channels * 2

        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            **kwargs,
        )


