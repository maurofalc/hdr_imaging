"""HDR-GAN: single-exposure LDR -> HDR reconstruction with a GAN + attention.

Small helper package shared by the ablation notebooks. The notebooks stay the primary
interface; this package only holds the reusable pieces (data loading, model builders,
losses/metrics) so they are defined once instead of copy-pasted per notebook.
"""

from .data import DEFAULT_TEST_INDICES, load_sice
from .losses import (
    ADV_WEIGHT,
    adversarial_loss,
    content_loss,
    discriminator_loss,
    generator_loss,
    psnr,
    ssim,
)
from .models import (
    SelfAttention,
    build_discriminator,
    build_generator,
    channel_attention,
)

__all__ = [
    "load_sice",
    "DEFAULT_TEST_INDICES",
    "build_generator",
    "build_discriminator",
    "channel_attention",
    "SelfAttention",
    "discriminator_loss",
    "adversarial_loss",
    "content_loss",
    "generator_loss",
    "psnr",
    "ssim",
    "ADV_WEIGHT",
]
