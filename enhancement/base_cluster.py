import json
import os 
import sys
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from lavis.models import load_model
from lavis.common.dist_utils import get_rank, init_distributed_mode

class SFTCluster:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sft_data_path = cfg.dataset.get("sft_data_path","")
        self.device = torch.device(f"cuda:{cfg.run.device}" if torch.cuda.is_available() else "cpu")
        self.dataset = None
        # self.dataset = SFTDataset(self.sft_data_path)
        
        self.samples = list()
        self.sample_dataloader = None
        self.sample_ids = list()
        self.sample_embeddings =  list()
        self.experiment_name = self.cfg.run.get("experiment_name", self.sft_data_path.split('/')[-1].split('.')[0])
        self.experiment_path = os.path.join(cfg.run.experiment_raw_path, self.experiment_name)
        self.sample_embeds_save_path = os.path.join(self.experiment_path,'sample_embeds')

        self.model = None
        self.samples_to_cluster = dict() # {'sample_id':'cluster_id',...}
        self.cluster_specific_path = os.path.join(self.experiment_path, cfg.run.type)
        self.split_samples_path = os.path.join(self.cluster_specific_path, 'split_samples')
        self.samples_to_cluster_file = os.path.join(self.cluster_specific_path,'samples_to_cluster.json')
        
        self.dis_center_points = None
        self.dis_center_points_save_path = os.path.join(self.cluster_specific_path,'distribution_center_points.pth')

    def setup_seeds(self):
        seed = self.cfg.run.seed + get_rank()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        cudnn.benchmark = False
        cudnn.deterministic = True

    def init_paths(self):
        # init cluster experiment paths
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)
        if not os.path.exists(self.cluster_specific_path):
            os.mkdir(self.cluster_specific_path)
        if not os.path.exists(self.sample_embeds_save_path):
            os.mkdir(self.sample_embeds_save_path)
        if not os.path.exists(self.split_samples_path):
            os.mkdir(self.split_samples_path)
    
    def print_key_config(self):
        print(" ------------- CONFIG ------------- ")
        print("sft_data_path: ", self.sft_data_path)
        print("device: ", self.device)
        print("experiment path: ", self.experiment_path)
        print("cluster specific path: ", self.cluster_specific_path)
        print("sample embedding save path: ", self.sample_embeds_save_path)
        print("split samples path: ", self.split_samples_path)

    def create_dataloader(self, batch_size):
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        return dataloader

    def load_data(self):
        pass
    
    def init_model(self):
        print(f"Loading encoding model to device {self.device} ...,\nmodel arch: {self.cfg.model.arch}, model type: {self.cfg.model.model_type}")
        self.model = load_model(
            self.cfg.model.arch,
            self.cfg.model.model_type, 
            is_eval=True, 
            device=self.device
        ) # BLIP2 first-stage model with Q-former and ViT.
   
    def calculate_embedding(self, sample):
        # 取 cls 位置的 embedding
        text = sample['text_input'] # currently cluster questions only
        text_inputs = self.model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.run.max_length, 
            return_tensors="pt",
        ).to(self.device)
        text_feats = self.model.forward_text(text_inputs)
        text_embeds = F.normalize(self.model.text_proj(text_feats))
        return text_embeds

    def encode(self):
        pass
