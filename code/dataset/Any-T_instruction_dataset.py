import json
import os.path

from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import re
import random
import numpy as np
import torch
from model.ImageBind import *


# from .base_dataset import BaseDataset


class Any2TInstructionDataset(Dataset):
    """
    Any - T instruction Dataset
    """
    def __init__(self, data_path: str, mm_root_path: str = None, dataset_type: str = 'AnyToText'):
        super(Any2TInstructionDataset, self).__init__()

        self.mm_root_path = mm_root_path
        self.inputs = []
        self.instructions = []
        self.outputs = []
        self.mm_names = []
        self.mm_paths = []
        self.device = 'cpu'
        
        with open(data_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
        
        for instance in tqdm(res, total=len(res)):
            self.inputs.append(instance['input'])
            self.instructions.append(instance['instruction'])
            self.outputs.append(instance['output'])
            self.mm_names.append(instance['mm_names'])
            self.mm_paths.append(instance['mm_paths'])
        
        self.dataset_type_list = [dataset_type for _ in range(len(self.inputs))]

    def __len__(self):  # number of instances
        return len(self.inputs)

    def __getitem__(self, i):
        mm_inputs = []
        
        if self.mm_names:
            for name, filepath in zip(self.mm_names[i], self.mm_paths[i]):
                path = os.path.join(self.mm_root_path, filepath)
                if name == "video":
                    mm_input = data.load_and_transform_video_data([path], self.device)
                if name == "audio":
                    mm_input = data.load_and_transform_audio_data([path], self.device)
                if name == "image":
                    mm_input = data.load_and_transform_vision_data([path], self.device)
                mm_inputs.append(mm_input[0])
        
        return dict(inputs=self.inputs[i], instructions=self.instructions[i],
                    outputs=self.outputs[i], mm_inputs=mm_inputs,
                    dataset_types=self.dataset_type_list[i])

