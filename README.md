# What's New in This Repository
* Vendored and patched Mask R-CNN (`mrcnn/`) directly into this repository -- no
  separate Mask-RCNN-TF2 clone step needed anymore.
* Updated the installation process for Python 3.11 / TensorFlow 2.15.
* Fixed correctness bugs in `grain_metrics.py` (see [PR #5](https://github.com/axkusakin/crop_seed_instance_segmentation/pull/5)).
* Included code for calculating seed morphological parameters.

## How-to
**1. Clone the repository:**
  ```
  git clone https://github.com/axkusakin/crop_seed_instance_segmentation.git
  cd crop_seed_instance_segmentation
  ```
**2. Set Up the Environment (Python 3.11 required)**

  ```
  mamba create -n <env_name> python=3.11
  conda activate <env_name>
  ```
**3. Install Dependencies**

  ```
  pip install -r requirements.txt
  ```

`mrcnn/` (Mask R-CNN) is vendored in this repository as a patched fork of
[ahmedfgad/Mask-RCNN-TF2](https://github.com/ahmedfgad/Mask-RCNN-TF2) (itself a
TF2 port of [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)),
patched to run under modern `tf.keras` (TensorFlow 2.15). See
[`mrcnn/PATCHES.md`](./mrcnn/PATCHES.md) for exactly what was changed and why.
No separate clone or `python setup.py install` step is required -- it imports
directly as `mrcnn.model`, `mrcnn.config`, `mrcnn.utils`, etc. as long as you
run scripts/notebooks from the repository root.

**Verified working combination:** Python 3.11.15, TensorFlow 2.15.0, Keras
2.15.0, NumPy 1.26.4. NumPy must stay below 2.0 -- TensorFlow 2.15 was built
against the NumPy 1.x ABI and fails to import under NumPy 2.x. If you add or
upgrade dependencies, re-verify `numpy==1.26.4` is still what gets installed.

Only the inference path (`mrcnn.model.MaskRCNN(mode="inference")`, `.detect()`,
`.load_weights()`) has been tested against this pin, matching how
`grain_metrics.py` and `Mask_RCNN.ipynb` use it. The training-only code paths
(`.compile()`, `.train()`, `model_temp.py` -- which was dropped as unused dead
code) were not exercised or fixed and should be assumed broken until someone
needs them.



# (REDAME from the original repository) Learning from Synthetic Dataset for Crop Seed Instance Segmentation

![image-20191204160204190](README.assets/image-20191204160204190.png)

**Overview of the proposed training process of crop seed instance segmentation.**



See https://www.biorxiv.org/content/10.1101/866921v2 for details



## Data included in this repository

- Codes in Jupyter Notebook format

  - [Mask RCNN Inference of crop seed images](./Mask_RCNN.ipynb)

  - [Multivariate Analysis and Visualization](multivariate_analysis.ipynb)

    

## Large Files are stored in Google Drive

https://drive.google.com/file/d/1g8bg9ter9DlKWgs0lfPZMQemRlzRVOQr/view?usp=sharing



### Contents

- Barley data
  - Synthetic Images and Masks of Test Data
  - Real World Images of Test Data (19 barley cultivar)
    - The annotation of Real World Images formated in JSON
  - Trained Model Weights
- Other crops
  - Model Weights and Image of Rice seeds
  - Model Weights and Images of 4 Wheat cultivars. One model can infer 4.



## Howto

1. Clone the repository

2. Install Dependencies (See below)

3. Download the data.zip from google drive and place it into the top directory of this repository

4. Run the notebook



## Dependencies

- [Mask RCNN](https://github.com/matterport/Mask_RCNN) implemented with Keras/Tensorflow provided by matterport.
- Keras==2.2.4
- Tensorflow-gpu==1.13.1
- pyefd==1.4.1 (for EFD analysis)
- other general packages such as sklearn, scikit-image, opencv3, etc..



## Author

Yosuke Toda

JST PRESTO / ITbM, Nagoya Univ.

