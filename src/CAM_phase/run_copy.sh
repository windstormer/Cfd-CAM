#! /bin/bash
if [ -n "$1" ]; then
    if [ "$1" == "t1" ]; then
        python3 main.py --pretrained_path Res50_t1_ep10_b256.Supcon -m t1 -c CfdCAM
        python3 main.py --pretrained_path Res50_t1_ep10_b256.Supcon -m t1 -c ScoreCAM
        python3 main.py --pretrained_path Res50_t1_ep10_b256.Supcon -m t1 -c CAM

    elif [ "$1" == "t2" ]; then
        python3 main.py --pretrained_path Res50_t2_ep10_b256.Supcon -m t2 -c CfdCAM
        python3 main.py --pretrained_path Res50_t2_ep10_b256.Supcon -m t2 -c CAM
        python3 main.py --pretrained_path Res50_t2_ep10_b256.Supcon -m t2 -c ScoreCAM

    elif [ "$1" == "t1ce" ]; then
        python3 main.py --pretrained_path Res50_t1ce_ep10_b256.Supcon -m t1ce -c CfdCAM
        python3 main.py --pretrained_path Res50_t1ce_ep10_b256.Supcon -m t1ce -c CAM
        python3 main.py --pretrained_path Res50_t1ce_ep10_b256.Supcon -m t1ce -c ScoreCAM

    elif [ "$1" == "flair" ]; then
        python3 main.py --pretrained_path Res50_flair_ep10_b256.Supcon -m flair -c CfdCAM
        python3 main.py --pretrained_path Res50_flair_ep10_b256.Supcon -m flair -c CAM
        python3 main.py --pretrained_path Res50_flair_ep10_b256.Supcon -m flair -c ScoreCAM

    else
    echo "Error modality. Usage: run.sh [modality]"

    fi
else
    echo "Empty modality. Usage: run.sh [modality]"
fi