# ADPA: Affective Dual-Path Adaptation for Parameter-Efficient Video Emotion Recognition

This repository provides the implementation of **ADPA (Affective Dual-Path Adaptation)** for parameter-efficient video emotion recognition.

ADPA is built on **UniFormerV2-B/16** and introduces two lightweight adaptation modules into the local Transformer blocks:

* **AAAdapter (Affective Attention Adapter)**: inserted into the attention path for low-rank semantic adaptation, lightweight temporal modeling, and input-aware coordination.
* **APAdapter (Affective Perceptron Adapter)**: inserted into the feed-forward path for lightweight low-rank feature adaptation.

During fine-tuning, the pretrained backbone is frozen, while the proposed adapters and classification head are optimized for video emotion recognition.

## Environment

This project is implemented based on **MMAction2**.

Please install the environment following the official MMAction2 installation guide:

[MMAction2 Installation Guide](https://github.com/open-mmlab/mmaction2/blob/main/docs/en/get_started/installation.md)

Main dependencies include:

* Python
* PyTorch
* MMEngine
* MMCV
* MMAction2
* Einops

## Pretrained Weights

We initialize the backbone using the official **UniFormerV2-B/16** checkpoint pretrained on **Kinetics-710** provided by MMAction2.

|Dataset|Uniform Sampling|Resolution|Backbone|Pretrain|Top-1 Acc|Top-5 Acc|
|-|-:|-|-|-|-:|-:|
|Kinetics-710|8|Raw|UniFormerV2-B/16\*|CLIP|78.9|94.2|

* [Download UniFormerV2-B/16 Kinetics-710 checkpoint](https://download.openmmlab.com/mmaction/v1.0/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb_20230612-63cdbad9.pth)
* [Official MMAction2 UniFormerV2 config](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/uniformerv2-base-p16-res224_clip-pre_u8_kinetics710-rgb.py)
* [MMAction2 UniFormerV2 Model Zoo](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/README.md)

Please download the checkpoint and set the corresponding pretrained-weight path in your configuration file before training.

## Datasets

Experiments are conducted on **Ekman-6** and **eMotions**.

### Ekman-6

Ekman-6 contains six basic emotion categories:

* Anger
* Disgust
* Fear
* Joy
* Sadness
* Surprise

Dataset access:

[Ekman-6 dataset (Zenodo)](https://zenodo.org/records/17159328)

The Zenodo record provides the raw videos and related resources for Ekman-6.

### eMotions

eMotions is a large-scale short-form video emotion recognition dataset containing six emotion categories:

* Excitation
* Fear
* Neutral
* Relaxation
* Sadness
* Tension

Only the **visual modality** is used in our experiments.

Dataset access:

* [eMotions dataset on Hugging Face](https://huggingface.co/datasets/Conna/eMotions)
* [Official eMotions repository](https://github.com/XuecWu/eMotions)

Please download the datasets separately and modify the dataset paths in the corresponding MMAction2 configuration files.

## Training

The ADPA model file is located at:

```text
ADPA/mmaction2-main/mmaction/models/backbones/ADPA.py
```

The configuration file is located at:

```text
./configs/recognition/uniformerv2/uniformerv2-base-p16-res224\_clip\_8xb32-u8\_kinetics400-rgb.py
```

Before training, please update the pretrained weight path and dataset path in the configuration file.

The model can be trained with the following commands:

```bash
export CUBLAS\_WORKSPACE\_CONFIG=":4096:8"

CUDA\_VISIBLE\_DEVICES=0,1 bash ./tools/dist_train.sh  ./configs/recognition/uniformerv2/uniformerv2-base-p16-res224\_clip\_8xb32-u8\_kinetics400-rgb.py  2  --work-dir ./workdir
```

## Testing

After downloading the fine-tuned ADPA model weights, you can reproduce the results reported in the paper using the following evaluation commands. Please modify the checkpoint path if necessary.

### Ekman-6

```bash
CUDA\_VISIBLE\_DEVICES=0,1 bash ./tools/dist_test.sh  ./configs/recognition/uniformerv2/uniformerv2-base-p16-res224\_clip\_8xb32-u8\_kinetics400-rgb.py  /ADPA/Ekman-6.pth  2
```

### eMotions

```bash
CUDA\_VISIBLE\_DEVICES=0,1 bash ./tools/dist_test.sh  ./configs/recognition/uniformerv2/uniformerv2-base-p16-res224\_clip\_8xb32-u8\_kinetics400-rgb.py  /ADPA/eMotions.pth  2
```

## Acknowledgements

This implementation is developed based on:

* [MMAction2](https://github.com/open-mmlab/mmaction2)
* [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2)

We sincerely thank the authors and contributors of these open-source projects.

