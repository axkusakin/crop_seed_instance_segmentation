# Patches applied to vendored Mask R-CNN

Source: [ahmedfgad/Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2)
(commit `e20bc04`, last upstream release Jan 2021, TF2 port of
[matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)). Upstream is
unmaintained (0 open PRs, last release 2021) and only tested against
TF 2.0.0 / Keras 2.2.4-2.3.1 / Python 3.6-3.7.

Target pin for this fork: **Python 3.11, TensorFlow 2.15.0, Keras 2.15.0,
NumPy 1.26.4** (empirically verified: clean install + import + model build +
`detect()` + `save_weights()`/`load_weights()` all work).

## Scope: inference only

`grain_metrics.py` and `Mask_RCNN.ipynb` only ever construct
`MaskRCNN(mode="inference")` and call `.detect()` / `.load_weights()`. They
never call `.compile()`, `.train()`, or `load_image_gt(..., augmentation=...)`.
All patches below were scoped and tested against that usage. Training-only
code paths (`compile()`'s `metrics_tensors` usage, `imgaug`-based augmentation
in `load_image_gt`, `parallel_model.py`'s multi-GPU wrapper) still contain
dead/untested legacy API calls and should be assumed broken until someone
actually exercises them.

`model_temp.py` (an unused near-duplicate of `model.py`, never imported by
anything in this repo) was dropped rather than vendored or patched.

## Files changed and why

### `model.py`
1. **`import keras` / `keras.backend` / `keras.layers` / `keras.models`** kept
   as standalone `keras` imports (not `tensorflow.keras`) -- at the TF 2.15
   pin, the standalone `keras` package (v2.15.0) *is* the same code TF ships,
   and unlike `from tensorflow import keras` it exposes `keras.__version__`,
   which `model.py`'s own version-assert relies on.
2. **`import keras.engine as KE` removed.** `keras.engine` no longer exists as
   an importable module in modern Keras. The only thing ever pulled from it
   was `KE.Layer`, used as the base class for `ProposalLayer`,
   `PyramidROIAlign`, `DetectionTargetLayer`, and `DetectionLayer`. Replaced
   with `KE = KL` (`keras.layers`), which has `Layer`.
3. **`from keras.engine import saving` / `from keras.engine import topology as
   saving`** (inside `load_weights()`) no longer exist either. Replaced with
   `from tensorflow.python.keras.saving import hdf5_format as saving` (falls
   back to `from keras.src.saving.legacy import hdf5_format as saving` if the
   first import path moves in a future Keras version). Both
   `load_weights_from_hdf5_group_by_name` and `load_weights_from_hdf5_group`
   exist on this module and were verified with a real save/load round-trip.
4. **`mask.astype(np.bool)` -> `mask.astype(bool)`.** `np.bool` was removed in
   NumPy >=1.24.
5. **`s = K.int_shape(x); mrcnn_bbox = KL.Reshape((s[1], ...))`
   (`fpn_classifier_graph`)**: under `tf.keras`'s stricter static shape
   inference, `s[1]` (`num_rois`) frequently resolves to `None` where old
   standalone Keras returned a concrete int, and `Reshape` rejects a `None`
   dimension (`ValueError: ... None values not supported`). This is a
   well-documented issue in every TF2 port of this codebase (see
   [matterport/Mask_RCNN#1820](https://github.com/matterport/Mask_RCNN/issues/1820),
   [#1070](https://github.com/matterport/Mask_RCNN/issues/1070)). Fixed by
   falling back to `-1` (runtime-inferred) when `s[1] is None`.
6. **`indices = tf.stack([tf.range(probs.shape[0]), ...])`
   (`refine_detections_graph`)**: same root cause -- `probs.shape[0]` (static)
   can be `None` under `tf.keras` graph tracing. Replaced with
   `tf.range(tf.shape(probs)[0])` (dynamic shape), which works in both eager
   and graph mode. Community-documented alongside the Reshape fix above.
7. **`if os.name is 'nt':` -> `if os.name == 'nt':`** (`SyntaxWarning: "is"
   with a literal` under Python >=3.8, since string literal identity is not
   guaranteed by the language spec).

### `utils.py`
- `np.bool` -> `bool` (4 occurrences: `mini_mask`/`mask` `.astype(np.bool)`
  calls and one `dtype=np.bool` array constructor).

### `parallel_model.py`
- `import keras.backend/layers/models` -> `from tensorflow.keras import ...`.
  This module (multi-GPU training wrapper) is only imported lazily inside
  `MaskRCNN.compile()`'s multi-GPU branch, so it is not exercised by this
  repo's inference-only usage, but was patched for import-cleanliness anyway
  since the fix was trivial and low-risk.

### `grain_metrics.py` (repo root, not in `mrcnn/`)
- Removed the dead `import keras.backend as K` (never referenced elsewhere in
  the file).
- No `sys.path` manipulation needed: `mrcnn/` lives at the repo root next to
  `grain_metrics.py`, so it's importable automatically when the script is run
  directly.

### `Mask_RCNN.ipynb`
- Removed the `git clone https://github.com/ahmedfgad/Mask-RCNN-TF2.git` cell
  and the `sys.path.append("Mask_RCNN")` line -- no longer needed now that
  `mrcnn/` is vendored in-repo.
- `.astype(np.int)` -> `.astype(int)` (`np.int` removed in NumPy >=1.24).

## What was checked and found *not* to be a problem

An earlier scoping pass estimated ~20 TF1-era calls (`tf.log`, `tf.sets.
set_intersection`, `tf.to_float`/`tf.to_int32`, `control_flow_ops`, `tf.
variable_scope`, `tf.get_variable`, bare `placeholder`). A direct grep across
the actual vendored source found **none of these patterns present** -- that
estimate was wrong. The real legacy-API surface was limited to what's listed
above. `imgaug` is imported only inside `load_image_gt`'s `if augmentation:`
branch, which nothing in this repo's actual call sites triggers, so it is not
a hard runtime dependency (documented as optional in `requirements.txt`).
