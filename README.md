# A Novel Confidence Induced Class Activation Mapping for MRI Brain Tumor Segmentation (Cfd-CAM) [Arxiv]
Official code implementation for the Cfd-CAM paper published on [arxiv](https://arxiv.org/abs/2306.05476).

This repository also include the implementation of [CAM](https://arxiv.org/abs/1512.04150) and [ScoreCAM](https://arxiv.org/abs/1910.01279).

## Dataset
[RSNA-ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2021](http://braintumorsegmentation.org/)
Download the official BraTS 2021 Dataset Task 1
Split the official training set into training and validation with the ratio 9:1.
The case id for training and validation set are shown in dataset.txt.
Preprocess the dataset from 3D volume data into 2D slide with the following script.
```
cd ./src/
python3 gen_dataset.py -m t1 -d training/validate
```

Folder Structures for Dataset
```
DATASET_NAME
|-- flair
|   |-- training
|   |   |-- normal
|   |   |   |-- NORMAL_1.png
|   |   |   |-- ...
|   |   |-- seg
|   |   |   |-- TUMOR_1.png
|   |   |   |-- ...
|   |   |-- tumor
|   |   |   |-- TUMOR_1.jpg
|   |   |   |-- ...
|   |-- validate
|   |   |-- normal
|   |   |   |-- NORMAL_1.png
|   |   |   |-- ...
|   |   |-- seg
|   |   |   |-- TUMOR_1.png
|   |   |   |-- ...
|   |   |-- tumor
|   |   |   |-- TUMOR_1.jpg
|   |   |   |-- ...
|-- t1
|-- t1ce
|-- t2
```
## Encoder Pretrain with Self-supervised Methods
```
cd ./src/model_phase/
python3 pretrain_clnet.py -m t1 --method_type Supcon --model_type Res18
```
## Train and Test Classifier with Pretrained Encoder
```
cd ./src/model_phase/
python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path SimCLR/Res18_t1_ep100_b512
python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256
```
## Run single-scale Cfd-CAM/CAM/ScoreCAM
```
cd ./src/CAM_phase/
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CfdCAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c ScoreCAM
```

## Run multi-scale Cfd-CAM/CAM/ScoreCAM
```
cd ./src/CAM_phase_ms_test_plus/
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CfdCAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c ScoreCAM
```

## Citation
If you use the code or results in your research, please use the following BibTeX entry.
```
@article{chen2023novel,
  title={A Novel Confidence Induced Class Activation Mapping for MRI Brain Tumor Segmentation},
  author={Chen, Yu-Jen and Shi, Yiyu and Ho, Tsung-Yi},
  journal={arXiv preprint arXiv:2306.05476},
  year={2023}
}
```
