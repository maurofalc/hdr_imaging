"""Loading the SICE dataset for LDR -> HDR reconstruction.

The dataset (Cai et al., 2018, https://github.com/csjcai/SICE) ships as two parts:

    <data_root>/Part1/Part1/<scene>/<exposure>.JPG   +   Part1/Part1/Label/<scene>.JPG
    <data_root>/Part2/Part2/<scene>/<exposure>.JPG   +   Part2/Part2/Label/<scene>.{JPG,PNG}

Part 1 has 360 scenes, Part 2 has 229 (589 total). Each scene is a bracketed exposure
sequence ordered dark -> bright; ``1.*`` is the most under-exposed frame and is used as
the LDR input, while ``Label/<scene>.*`` is the ground-truth HDR reconstruction.

Two things that broke the original notebooks are handled here:

* **Case / extension sensitivity.** Files are ``.JPG`` on disk and Part 2 labels are a
  mix of ``.JPG`` and ``.PNG``. ``cv2.imread(".../1.jpg")`` happens to work on Windows
  (case-insensitive filesystem) but fails on Colab/Linux. Files are resolved with a
  case- and extension-agnostic glob instead of a hard-coded name.
* **Data leakage.** The original U-Net / plain-GAN notebooks reused scenes 357-360 in
  both training and test. Here a fixed set of held-out scenes is *removed* from the
  training pool (the split from the thesis' attention notebooks), and ``load_sice``
  asserts the two sets are disjoint before returning.
"""

import glob
import os

import cv2
import numpy as np

# Held-out test scenes (1-based positions in the concatenated 589-scene list), taken
# from the thesis' attention-gan / spatial-gan notebooks. Kept in descending order so
# that popping them one by one never shifts a position still to be removed.
DEFAULT_TEST_INDICES = [103, 100, 79, 75, 69, 55, 52, 46, 40, 39, 37, 34, 33, 31, 28, 23, 4]

# (folder base name, number of scenes) in the order the thesis concatenated them.
_PARTS = [("Part1", 360), ("Part2", 229)]


def _resolve_part_dir(data_root, base):
    """Find the nested part directory, tolerating a couple of known namings."""
    candidates = [
        os.path.join(data_root, base, base),                # Part1/Part1  (as on disk)
        os.path.join(data_root, f"Dataset_{base}", f"Dataset_{base}"),  # original notebooks
        os.path.join(data_root, base),                      # flat layout
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{base}' under {data_root!r}. Tried: {candidates}"
    )


def _find_file(directory, stem):
    """Return the file in ``directory`` whose name (without extension) equals ``stem``,
    ignoring case and extension. Returns ``None`` if nothing matches."""
    stem = str(stem).lower()
    for path in glob.glob(os.path.join(directory, "*")):
        base, _ext = os.path.splitext(os.path.basename(path))
        if base.lower() == stem:
            return path
    return None


def _load_image(path, size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


def load_sice(data_root, size=512, test_indices=None, normalize=True,
              limit_per_part=None, verbose=True):
    """Load SICE into ``(x_train, y_train), (x_test, y_test)`` numpy arrays.

    Args:
        data_root: directory containing ``Part1`` and ``Part2``.
        size: images are resized to ``(size, size)``.
        test_indices: 1-based held-out positions; defaults to ``DEFAULT_TEST_INDICES``.
        normalize: divide pixel values by 255 into ``[0, 1]``.
        limit_per_part: cap scenes per part (handy for quick smoke tests).
        verbose: print a short summary / warn on missing scenes.
    """
    if test_indices is None:
        test_indices = list(DEFAULT_TEST_INDICES)

    samples, gts = [], []
    for base, n_scenes in _PARTS:
        part_dir = _resolve_part_dir(data_root, base)
        label_dir = os.path.join(part_dir, "Label")
        n = n_scenes if limit_per_part is None else min(n_scenes, limit_per_part)
        for scene in range(1, n + 1):
            input_path = _find_file(os.path.join(part_dir, str(scene)), 1)
            label_path = _find_file(label_dir, scene)
            if input_path is None or label_path is None:
                if verbose:
                    print(f"  [skip] {base} scene {scene}: input or label missing")
                continue
            samples.append(_load_image(input_path, size))
            gts.append(_load_image(label_path, size))

    if len(set(test_indices)) != len(test_indices):
        raise ValueError("test_indices contains duplicates")
    if max(test_indices) > len(samples):
        raise ValueError(
            f"test index {max(test_indices)} exceeds the {len(samples)} scenes loaded "
            f"(did you set limit_per_part too low?)"
        )

    # Remove the held-out scenes from the training pool (descending order => safe pops).
    test_samples, test_gts = [], []
    for idx in test_indices:
        test_samples.append(samples.pop(idx - 1))
        test_gts.append(gts.pop(idx - 1))

    def _stack(arr):
        out = np.asarray(arr, dtype="float32")
        return out / 255.0 if normalize else out

    x_train, y_train = _stack(samples), _stack(gts)
    x_test, y_test = _stack(test_samples), _stack(test_gts)

    # Leakage guard: training and test pools must not overlap.
    assert x_train.shape[0] + x_test.shape[0] == len(samples) + len(test_samples)
    assert x_test.shape[0] == len(test_indices)

    if verbose:
        print(f"train: {x_train.shape}  test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)
