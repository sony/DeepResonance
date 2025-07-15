# DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning
- Paper: [arxiv](https://arxiv.org/abs/2502.12623)
- This codebase was built upon and extended from the [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT) framework.

## Setup
```
pip install -r requirements.txt
```
Or you can use ```Dockerfile``` to establish a virtual environment.

## Prepare Pre-trained Model Checkpoints
- Put [DeepResonance_α](https://huggingface.co/datasets/Sony/DeepResonance_data_models/tree/main/ckpt) and [DeepResonance_β](https://huggingface.co/datasets/Sony/DeepResonance_data_models/tree/main/ckpt) downloaded from Huggingface into ```./ckpt/```
- Referring to [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT?tab=readme-ov-file#Prepare%20Pre-trained%20Checkpoint), please prepare the checkpoints of **ImageBind (huge)** and **Vicuna (7b-v0)** in ```./ckpt/pretrained_ckpt```

## Inference with Deepresonance-α
```
cd code
$DATASET=musiccaps # can replace with others, dataset path is defined in inference_deepresonance.py; select from [musicqa, musiccaps, music4way_musiccaps, music4way_mi2t, music4way_mv2t, music4way_any2t]
$OUTPUT=musiccaps_dra_res
python inference_deepresonance.py --dataset $DATASET --result_file_name $OUTPUT --ckpt_path ../ckpt/deepresonance_alpha_delta_ckpt/deepresonance/7b_tiva_v0 --imagebind_embs_seq
```

## Inference with Deepresonance-β
```
cd code
$DATASET=musiccaps # can replace with others, dataset path is defined in inference_deepresonance.py; select from [musicqa, musiccaps, music4way_musiccaps, music4way_mi2t, music4way_mv2t, music4way_any2t]
$OUTPUT=musiccaps_drb_res
python inference_deepresonance.py --dataset $DATASET --result_file_name $OUTPUT --ckpt_path ../ckpt/deepresonance_beta_delta_ckpt/deepresonance/7b_tiva_v0 --prellmfusion --imagebind_embs_seq
```

## Prepare Data
- Put all the text [datasets](https://huggingface.co/datasets/Sony/DeepResonance_data_models/tree/main/data) downloaded from Huggingface into ```./data```
- For multimodal source data including music, videos, and images, they should be downloaded separately with the IDs shown in each text file. We do not provide due to the licence issues. Please download them with the IDs; all the data is originally from the AudioSet dataset. Refer to [the filtered subset of M2UGen and download all the video and music pairs from YouTube](https://github.com/sdpigpig/M2UGen/blob/main/Datasets/common/filtered.csv)

## Model Training
- Train DeepResonance-α:
```
cd code
source scripts/train_deepresonance_alpha.sh ../ckpt/deepresonance_alpha_delta_ckpt_exp1
```

- Train DeepResonance-β:
```
cd code
source scripts/train_deepresonance_beta.sh ../ckpt/deepresonance_beta_delta_ckpt_exp1
```

## Cite
If you find this repo useful, please consider citing:
```bibtex
@article{DBLP:journals/corr/abs-2502-12623,
  author       = {Zhuoyuan Mao and
                  Mengjie Zhao and
                  Qiyu Wu and
                  Hiromi Wakaki and
                  Yuki Mitsufuji},
  title        = {DeepResonance: Enhancing Multimodal Music Understanding via Music-centric
                  Multi-way Instruction Tuning},
  journal      = {CoRR},
  volume       = {abs/2502.12623},
  year         = {2025},
  url          = {https://doi.org/10.48550/arXiv.2502.12623},
  doi          = {10.48550/ARXIV.2502.12623},
  eprinttype    = {arXiv},
  eprint       = {2502.12623},
  timestamp    = {Wed, 19 Mar 2025 11:49:47 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2502-12623.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

