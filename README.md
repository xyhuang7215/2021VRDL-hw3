## VRDL HW3

## Colab
You may use [Colab notebook](https://colab.research.google.com/drive/1cenS7LFju9EOAWr0CXsDOkzNVjyM2XIn?usp=sharing)(recommend) to get the inference result, or follow the instruction below.

## Requirements

Install [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetecion](https://github.com/open-mmlab/mmdetection)

## Dataset & annotation & config
Unzip the files in mmdetection
- [Download](https://drive.google.com/file/d/1HrEgaGQkaFVHMvN9WjcjN_jXE5DxXYlg/view?usp=sharing)

## Training

```train
cd mmdetection
python tools/train.py dataset/mask_rcnn_50.py
```

## Inference

```eval
python inference.py 
```

## Pre-trained Models

You can download pretrained models here:

- [My model](https://drive.google.com/drive/folders/1-1pY0VQoljHAqa5YxDsi_ut6I_zLRmw9?usp=sharing)

