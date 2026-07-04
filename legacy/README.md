# Legacy notebooks (deprecated)

These are the **original** thesis notebooks, kept for historical reference. They are
superseded by the cleaned, standardized ablation notebooks (`01_unet` … `05_gan_full`) in
the repository root, which share the code in [`src/hdr_gan/`](../src/hdr_gan).

They are preserved as-is (including their original outputs) and are **not maintained**.
They contain the issues described in the main README's "What changed" section:

- `main.ipynb` — plain GAN. Test-set loading reused a stale loop variable, and the
  train/test split leaked (test images were a subset of the training images).
- `channel-gan.ipynb` — GAN + channel attention. Redefines the generator three times
  (only the last runs), with a dead channel-attention helper, an empty spatial-attention
  stub, and an abandoned spatial-attention attempt using deprecated `keras.backend` calls.
- `attention-gan.ipynb` — the original **full model** (channel + spatial attention), i.e.
  the thesis' actual proposal. It was never published; it is included here because
  `05_gan_full.ipynb` is its cleaned successor.

Prefer the root notebooks for anything other than looking at the original work.
