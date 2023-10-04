import argparse
import random
import os
import numpy as np
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS")
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from enhancement.base_cluster import SFTCluster

from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.metrics import pairwise_distances


class SFTInferenceDataset(Dataset):
    def __init__(self, vis_root=None, ann_path=None):
        self.annotation = list()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        return self.annotation[index]

class MMERawDataset(Dataset):
    def __init__(self, vis_root=None, ann_path=None, clean=True):
        self.tmp_annotation = json.load(open(ann_path, 'r'))
        self.annotation = list()
        test_split = ann_path.split('/')[-1].split('.')[0]
        cnt = 0
        for ann in self.tmp_annotation:
            for one_question in ann['questions']:
                question, label = one_question.split('\t')
                if clean:
                    question = question.replace('Please answer yes or no.','')
                image_path = '/'.join(ann['image'].split('/')[-2:])
                question_id = test_split + str(cnt).zfill(4)
                self.annotation.append({
                    'question_id': question_id,
                    'image_path': os.path.join(vis_root, image_path),
                    'text_input':question.strip(),
                    'text_output':label.strip(),
                    'fullAnswer': ann.get('fullAnswer',"")
                })
                cnt += 1

    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        return self.annotation[index]

class GQARawDataset(Dataset):
    def __init__(self, vis_root, ann_path):
        self.tmp_annotation = json.load(open(ann_path, 'r'))
        self.annotation = list()
        for ann in self.tmp_annotation:
            image_path = os.path.join(vis_root, ann["image"])
            self.annotation.append({
                "question_id":ann["question_id"],
                "image_path": image_path,
                "text_input": ann["question"],
                "text_output": ann["answer"],
                "fullAnswer": ann["fullAnswer"],
            }) 
    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, index):
        return self.annotation[index]

class ClusterInference(SFTCluster):
    def __init__(self, cfg):
        super().__init__(cfg)

    def load_center_points(self):
        print("Loading center points from : ", self.dis_center_points_save_path)
        self.dis_center_points = torch.load(self.dis_center_points_save_path)

    def encode(self, test_split, dataloader):
        # Done:
        # 1. use blip2 first-stage model to encode text(question)
        # 2. sample embeddings saved to split_embeds_save_path
        if self.cfg.run.re_encode:
            # mkdir embed save path
            split_embeds_save_path = os.path.join(self.sample_embeds_save_path, test_split)
            if not os.path.exists(split_embeds_save_path):
                os.mkdir(split_embeds_save_path)

            # calculate embedding for each sample in dataloader of each test_split
            for sample in tqdm(dataloader, desc=f"Encode {test_split}"):
                sample_ids = sample['question_id']
                text_embeds = self.calculate_embedding(sample)
                for sample_id, text_embed in zip(sample_ids, text_embeds):
                    torch.save(text_embed.to(torch.device('cpu')), os.path.join(split_embeds_save_path, f"{sample_id}.pth"))
            print(f"Sample embeddings save to : {split_embeds_save_path}")

    def kmeans_predict(self, text_embeds):
        ## predict
        # select the closest cluster center
        sample_dist = pairwise_distances(self.dis_center_points, np.array(text_embeds), metric='euclidean')
        select_index = list()
        for i in range(sample_dist.shape[1]):
            select_index.append(np.argmin(sample_dist[:,i]))
        return select_index

    def split_samples_by_sim(self, test_split, dataloader):
        split_embeds_save_path = os.path.join(self.sample_embeds_save_path, test_split)
        print("Loading sample embeddings from : ",split_embeds_save_path)

        samples_by_cluster = defaultdict(list)
        for sample in tqdm(dataloader, desc=test_split):
            sample_ids = sample['question_id']
            text_embeds = list()

            for sample_id in sample_ids:
                file_path = os.path.join(split_embeds_save_path, f"{sample_id}.pth")
                embed = torch.load(file_path)
                text_embeds.append(embed.detach().numpy())
            
            cluster_ids_y = self.kmeans_predict(text_embeds)

            for question_id, image_path, text_input, text_output, full_answer,cluster_id in zip(sample['question_id'], sample['image_path'], sample['text_input'], sample['text_output'], sample['fullAnswer'],cluster_ids_y):
                cluster = cluster_id.item()
                if self.cfg.dataset.mme_clean:
                    # remove instruction when encoding the question
                    # recover to the previous text input
                    text_input = text_input + ' Please answer yes or no.'
                tmp = {
                    'question_id': str(question_id),
                    'image_path': image_path,
                    'text_input': text_input,
                    'text_output': text_output,
                    'full_answer': full_answer,
                    'cluster': cluster
                }
                samples_by_cluster[cluster].append(tmp)

        statistic = dict()
        for k,v in samples_by_cluster.items():
            statistic[k] = len(v)

        specific_split_save_path = os.path.join(self.split_samples_path, test_split+'.json')
        print("Saving {} split to {}".format(test_split, specific_split_save_path))
        with open(specific_split_save_path, "w") as f:
            f.write(json.dumps(samples_by_cluster))
        
        return {test_split : statistic}


class MMEClusterInference(ClusterInference):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.init_paths()
        self.MME_ROOT = self.cfg.dataset.vis_root
        self.SPLIT_LIST = self.cfg.dataset.test_list
        self.SPLIT_MAP = {
            split.split('.')[0]: split for split in self.SPLIT_LIST
        }
        self.sample_embeds_save_path = self.cfg.run.get("sample_embeds_save_path", self.sample_embeds_save_path)
        print(self.sample_embeds_save_path)
        self.dis_center_points_save_path = self.cfg.run.center_points_path

    def predict(self):
        # init
        self.load_center_points()  
        if self.cfg.run.re_encode:
            self.init_model()      

        # init MME split
        test_list = list(self.SPLIT_MAP.keys())
        statistics = list()

        # generate mme split by cluster
        for test_split in test_list:
            ann_path = self.SPLIT_MAP[test_split]
            ann_path = os.path.join(self.MME_ROOT, ann_path)

            self.dataset = MMERawDataset(vis_root=self.MME_ROOT,ann_path=ann_path, clean=self.cfg.dataset.mme_clean)
            dataloader = self.create_dataloader(batch_size=self.cfg.dataset.batch_size)

            self.encode(test_split, dataloader)
            statistic = self.split_samples_by_sim(test_split, dataloader)

            statistics.append(statistic)
            print(statistic)
        
        split_statistics_save_path = os.path.join(self.split_samples_path, 'statistics.json')
        with open(split_statistics_save_path, "w") as f:
            f.write(json.dumps(statistics))


class GQAClusterInference(ClusterInference):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.init_paths()
        self.ann_path = self.cfg.dataset.get("ann_path","/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/testdev_balanced_questions.json")
        self.vis_root = self.cfg.dataset.get("vis_root","/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/images/")
        self.dis_center_points_save_path = self.cfg.run.center_points_path

    def predict(self):
        # init
        self.load_center_points()  
        if self.cfg.run.re_encode:
            self.init_model()

        self.dataset = GQARawDataset(self.vis_root, self.ann_path)
        dataloader = self.create_dataloader(batch_size=self.cfg.dataset.batch_size)
        self.encode("gqa", dataloader)
        statistics = self.split_samples_by_sim("gqa", dataloader)
        print(statistics)

        split_statistics_save_path = os.path.join(self.split_samples_path, 'statistics.json')
        with open(split_statistics_save_path, "w") as f:
            f.write(json.dumps(statistics))

def main():
    # config_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/enhancement/config/gqa_inference_cluster.yaml"
    config_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/enhancement/config/mme_inference_cluster.yaml"

    cfg = OmegaConf.load(config_file)

    if "gqa_inference" in config_file:
        sdc = GQAClusterInference(cfg)
        sdc.predict()
    elif "mme_inference" in config_file:
        sdc = MMEClusterInference(cfg)
        sdc.predict()

if __name__ == '__main__':
    main()