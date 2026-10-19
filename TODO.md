# TODO / Roadmap

## Known issues

- [ ] **Weights are no longer freely downloadable.** The Google Drive file `data_downloader.py` pulls from now
  requires the owner to approve an access request, so `python data_downloader.py` fails for anyone who hasn't been
  granted access. Not a blocker for existing holders of the weights, but blocks first-time setup for anyone else who
  clones this repo. Plan: re-host/mirror the weights (and rest of `data.zip`) somewhere with open access.

## Planned features

- [ ] **More morphological metrics**: symmetry/skew index, elliptic Fourier descriptors (EFD). `pyefd` is already a
  pinned dependency, but EFD computation isn't wired into either script yet — `multivariate_analysis.ipynb` only
  loads *precomputed* EFD coefficients from a CSV folder that doesn't exist in this repo. New descriptors should hook
  into `seedseg/measure.py:measure_instances`, using the contour/hull/ellipse already computed there per instance.
- [ ] **More preprocessing steps**: flip, CLAHE contrast and Gaussian blur are in `seedseg/preprocess.py`;
  grayscale conversion is still open. Contrast/blur settings still need validating against reference masks.
- [ ] **Defective-grain detection**: either a post-hoc heuristic computed from the existing mask/contour/metrics, or
  retraining Mask R-CNN with a third ("defective") class. The training path (`.compile()`/`.train()`) is currently
  unverified and should be assumed broken until someone exercises it (see `mrcnn/PATCHES.md`).
- [ ] Committed test image set: 3 real barley scans (500 dpi, TIF originals converted to JPEG — pixel dimensions and
  DPI tag verified unchanged by the conversion) currently staged in `test_images/{tif,jpeg}/` one level up from this
  repo. Still needs to be moved into the repo and committed (via git-lfs, per `.gitattributes`) so others have
  something to validate changes against.
