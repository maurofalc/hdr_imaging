"""Loss functions and quality metrics for the HDR-GAN.

The generator is trained with a content loss (per-pixel MSE against the ground-truth
HDR image) plus a small adversarial term, following the thesis:

    L_G = L_content + adv_weight * L_adversarial      (adv_weight = 1e-3)

The content loss is a plain mean-squared error over every pixel and channel. The
original notebook divided the summed error by ``batch_size ** 2``, which made the loss
scale with the batch size; ``tf.reduce_mean`` removes that dependency.
"""

import tensorflow as tf

# Binary cross-entropy on probabilities (the discriminator ends in a sigmoid).
_bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

ADV_WEIGHT = 1e-3


def discriminator_loss(real_output, fake_output):
    """Discriminator wants real HDR -> 1 and generated -> 0."""
    real_loss = _bce(tf.ones_like(real_output), real_output)
    fake_loss = _bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def adversarial_loss(fake_output):
    """Generator wants the discriminator to label its output as real (1)."""
    return _bce(tf.ones_like(fake_output), fake_output)


def content_loss(hdr, generated):
    """Mean-squared error between the generated image and the ground-truth HDR."""
    return tf.reduce_mean(tf.square(hdr - generated))


def generator_loss(fake_output, hdr, generated, adv_weight=ADV_WEIGHT):
    """Full generator objective: content loss + weighted adversarial loss."""
    return content_loss(hdr, generated) + adv_weight * adversarial_loss(fake_output)


# --------------------------------------------------------------------------- #
# Metrics (images are in [0, 1])
# --------------------------------------------------------------------------- #
def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio, averaged over the batch."""
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))


def ssim(y_true, y_pred):
    """Structural similarity index, averaged over the batch."""
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
