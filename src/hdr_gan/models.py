"""Generator, discriminator and attention blocks for the HDR-GAN.

The architecture follows the thesis (a U-Net generator with an attention module at
the bottleneck, plus a simple CNN discriminator), but keeps the *networks themselves*
cleaner than the original notebooks:

* The decoder is a standard U-Net decoder (upsample once, concatenate the skip
  connection, then refine with a stride-1 conv). The original notebooks upsampled and
  downsampled inside every decoder block, which fought itself and wasted capacity.
* Transposed convolutions use a kernel size of 4 (divisible by the stride of 2) to
  reduce the checkerboard artefacts the thesis itself set out to avoid.
* Channel attention uses a proper shared MLP (CBAM style) instead of 3x3 convolutions
  applied to 1x1 pooled maps.
* Spatial attention is the thesis' self-attention block, implemented as a small Keras
  layer with a learnable residual scale (SAGAN style) so it is correct under TF2/Keras.

A single ``build_generator(use_channel, use_spatial)`` builds every ablation variant:

    U-Net / plain GAN generator : build_generator(False, False)
    GAN + channel attention     : build_generator(True,  False)
    GAN + spatial attention     : build_generator(False, True)
    Full attention (the model)  : build_generator(True,  True)
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential

DEFAULT_INPUT_SHAPE = (512, 512, 3)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
# LeakyReLU's slope argument is passed positionally on purpose: it is ``alpha`` in
# Keras 2 and ``negative_slope`` in Keras 3 (the default in TF 2.16+ / current Colab),
# but the first positional argument works on both, so the code runs either way.
def _conv_bn_lrelu(x, filters, kernel_size=4, strides=2):
    """Downsampling conv block: Conv2D -> BatchNorm -> LeakyReLU."""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(0.2)(x)
    return x


def _up_concat_refine(x, skip, filters):
    """Standard U-Net decoder stage: upsample x (2x), concat skip, refine (stride 1)."""
    x = layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.concatenate([x, skip], axis=-1)
    x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.LeakyReLU(0.2)(x)
    return x


def channel_attention(x, ratio=8):
    """CBAM-style channel attention with a shared MLP over global avg/max pooling."""
    channels = x.shape[-1]
    shared_1 = layers.Dense(channels // ratio, activation="relu")
    shared_2 = layers.Dense(channels)

    avg = layers.GlobalAveragePooling2D()(x)
    avg = shared_2(shared_1(avg))
    mx = layers.GlobalMaxPooling2D()(x)
    mx = shared_2(shared_1(mx))

    mask = layers.Activation("sigmoid")(layers.Add()([avg, mx]))
    mask = layers.Reshape((1, 1, channels))(mask)
    return layers.Multiply()([x, mask])


class SelfAttention(layers.Layer):
    """Spatial self-attention (thesis Fig. 6 / SAGAN) with a learnable residual scale."""

    def __init__(self, projection_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.projection_ratio = projection_ratio

    def build(self, input_shape):
        channels = int(input_shape[-1])
        proj = max(channels // self.projection_ratio, 1)
        self.f = layers.Conv2D(proj, 1, padding="same")   # query
        self.g = layers.Conv2D(proj, 1, padding="same")   # key
        self.h = layers.Conv2D(channels, 1, padding="same")  # value
        self.gamma = self.add_weight(
            name="gamma", shape=(1,), initializer="zeros", trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        shape = tf.shape(x)
        batch, height, width = shape[0], shape[1], shape[2]
        channels = x.shape[-1]
        proj = max(int(channels) // self.projection_ratio, 1)

        f = tf.reshape(self.f(x), [batch, height * width, proj])
        g = tf.reshape(self.g(x), [batch, height * width, proj])
        h = tf.reshape(self.h(x), [batch, height * width, int(channels)])

        attention = tf.nn.softmax(tf.matmul(g, f, transpose_b=True), axis=-1)
        out = tf.matmul(attention, h)
        out = tf.reshape(out, [batch, height, width, int(channels)])
        return self.gamma * out + x

    def get_config(self):
        config = super().get_config()
        config.update({"projection_ratio": self.projection_ratio})
        return config


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
def build_generator(use_channel=False, use_spatial=False,
                    input_shape=DEFAULT_INPUT_SHAPE, name=None):
    """U-Net generator with optional channel/spatial attention at the bottleneck."""
    inputs = layers.Input(shape=input_shape)

    # Encoder: 512 -> 256 -> 128 -> 64 -> 32 -> 16
    e1 = _conv_bn_lrelu(inputs, 64)    # 256
    e2 = _conv_bn_lrelu(e1, 128)       # 128
    e3 = _conv_bn_lrelu(e2, 256)       # 64
    e4 = _conv_bn_lrelu(e3, 512)       # 32
    bottleneck = _conv_bn_lrelu(e4, 512)  # 16

    # Attention module at the bottleneck
    if use_channel:
        bottleneck = channel_attention(bottleneck)
    if use_spatial:
        bottleneck = SelfAttention()(bottleneck)

    # Decoder: 16 -> 32 -> 64 -> 128 -> 256 -> 512
    d4 = _up_concat_refine(bottleneck, e4, 512)  # 32
    d3 = _up_concat_refine(d4, e3, 256)          # 64
    d2 = _up_concat_refine(d3, e2, 128)          # 128
    d1 = _up_concat_refine(d2, e1, 64)           # 256
    outputs = layers.Conv2DTranspose(
        input_shape[-1], 4, strides=2, padding="same", activation="sigmoid"
    )(d1)  # 512

    if name is None:
        name = "generator_{}{}".format(
            "c" if use_channel else "", "s" if use_spatial else ""
        ) or "generator_unet"
    return Model(inputs, outputs, name=name)


# --------------------------------------------------------------------------- #
# Discriminator
# --------------------------------------------------------------------------- #
def build_discriminator(input_shape=DEFAULT_INPUT_SHAPE):
    """Simple CNN classifier that scores an image as real (HDR) or fake (generated)."""
    return Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, 4, strides=2, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(0.2),
            layers.Conv2D(512, 4, strides=2, padding="same"),
            layers.BatchNormalization(momentum=0.8),
            layers.LeakyReLU(0.2),
            layers.Conv2D(1, 4, strides=1, padding="same"),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
