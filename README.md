# Cfd-CAM

## Encoder Pretrain with Self-supervised Methods
```
cd ./model_phase/
python3 pretrain_clnet.py -m t1 --method_type Supcon --model_type Res18
```
## Train and Test Classifier with Pretrained Encoder
```
cd ./model_phase/
python3 train_cnet.py -b 256 -m t1 --encoder_pretrained_path SimCLR/Res18_t1_ep100_b512
python3 test_cnet.py -m t1 --pretrained_path Res18_t1_ep10_b256
```
## Run single-scale Cfd-CAM/CAM/ScoreCAM
```
cd ./CAM_phase/
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CfdCAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c ScoreCAM
```

## Run multi-scale Cfd-CAM/CAM/ScoreCAM
```
cd ./CAM_phase_ms_test_plus/
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CfdCAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c CAM
python3 main.py --pretrained_path Res18_t1_ep10_b256.Supcon -m t1 -c ScoreCAM
```
