import os


class DatasetCatalog:
    def __init__(self, server):
        if server == "local":
            self.music4way_m2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_m2t.json",
                    mm_root_path="../data/music4way/train/audios",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_i2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_i2t.json",
                    mm_root_path="../data/music4way/train/images",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_v2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../music4way_v2t.json",
                    mm_root_path="../data/music4way/train/videos",
                    dataset_type="AnyToText",
                ),
            }
            self.coco = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/coco.json",
                    mm_root_path="../data/coco/images",
                    dataset_type="AnyToText",
                ),
            }
        else:
            raise ValueError

        if server == "local":
            self.alpaca = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/alpaca.json",
                    dataset_type="AnyToText",
                ),
            }
            self.musicqa_mtt = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/musicqa-mtt.json",
                    mm_root_path="../data/musicqa/audios",
                    dataset_type="AnyToText",
                ),
            }
            self.musiccaps = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/musiccaps.json",
                    mm_root_path="../data/musiccaps/audios",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_m2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_m2t.json",
                    mm_root_path="../data/music4way/train/audios",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_i2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_i2t.json",
                    mm_root_path="../data/music4way/train/images",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_v2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_v2t.json",
                    mm_root_path="../data/music4way/train/videos",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_mi2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_mi2t.json",
                    mm_root_path="../data/music4way/train",
                    dataset_type="AnyToText",
                ),
            }
            self.music4way_mv2t = {
                "target": "dataset.Any-T_instruction_dataset.Any2TInstructionDataset",
                "params": dict(
                    data_path="../data/music4way_mv2t.json",
                    mm_root_path="../data/music4way/train",
                    dataset_type="AnyToText",
                ),
            }
        else:
            raise ValueError

