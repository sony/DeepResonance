#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import os
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from .utils import process_caption
from model.ImageBind import *


class BaseDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, mm_root_path: str, embed_path: str, dataset_type: str):
        super(BaseDataset, self).__init__()
        self.embed_path = embed_path
        self.mm_path_list, self.caption_list = [], []
        self.dataset_type_list = []
        self.device = "cpu"

    def __len__(self):  # number of instances
        return len(self.mm_path_list)

    def __getitem__(self, i): # currently supports X2T and T2X types.

        if self.dataset_type_list[i] == "VideoToText":
            inputs = data.load_and_transform_video_data([self.mm_path_list[i]], self.device)
        elif self.dataset_type_list[i] == "AudioToText":
            inputs = data.load_and_transform_audio_data([self.mm_path_list[i]], self.device)
        elif self.dataset_type_list[i] == "ImageToText":
            inputs = data.load_and_transform_vision_data([self.mm_path_list[i]], self.device)
        
        if self.dataset_type_list[i].split("To")[1] == "Text":
            return dict(mm_inputs=inputs[0], output_texts=self.caption_list[i],
                    dataset_types=self.dataset_type_list[i])
        else:
            with open(os.path.join(self.embed_path, str(os.path.basename(self.mm_path_list[i])) + '.npy'), 'rb') as f:
                caption_embs = torch.from_numpy(np.load(f, allow_pickle=True))  # (num_clip_tokens, 768)
            return dict(output_texts=self.caption_list[i], caption_embs=caption_embs,
                    dataset_types=self.dataset_type_list[i])

"""
    def collate(self, instances):
        mm_paths, output_texts, caption_embs, dataset_types = tuple(
            [instance[key] for instance in instances] for key in
            ("mm_paths", "output_texts", "caption_embs", "dataset_types"))
        return dict(
            mm_paths=mm_paths,
            output_texts=output_texts,
            caption_embs=caption_embs,
            dataset_types=dataset_types
        )
"""
