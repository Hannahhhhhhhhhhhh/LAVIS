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
from enhancement.k_means import SFT_Data_Cluster
from kmeans_pytorch import kmeans, kmeans_predict

from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode


MME_ROOT = "/mnt/pfs-guan-ssai/nlu/dingyifeng/multimodal/MME_Benchmark_release"
SPLIT_LIST = [
    'artwork.json',
    'celebrity.json',
    'code_reasoning.json',
    'color.json',
    'commonsense_reasoning.json',
    'count.json',
    'existence.json',
    'landmark.json',
    'numerical_calculation.json',
    'OCR.json',
    'position.json',
    'posters.json',
    'scene.json',
    'text_translation.json',
]

SPLIT_MAP = {
    split.split('.')[0]: split for split in SPLIT_LIST
}

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path",
        required=False,
        help="path to configuration file.",
        default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/mme_zeroshot_flant5xxl_instructblip_eval.yaml",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args

class MMERawDataset(Dataset):
    def __init__(self, vis_root=None, ann_path=None):
        self.tmp_annotation = json.load(open(ann_path, 'r'))
        self.annotation = list()
        for ann in self.tmp_annotation:
            for one_question in ann['questions']:
                question, label = one_question.split('\t')
                image_path = '/'.join(ann['image'].split('/')[-2:])
                self.annotation.append({
                    'question_id': ann.get("image",""),
                    'image_path': os.path.join(vis_root, image_path),
                    'text_input':question.strip(),
                    'text_output':label.strip(),
                    'fullAnswer': ann.get('fullAnswer',"")
                })
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


def create_dataloader(dataset, batch_size=1):

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )
    return dataloader

def init_sft_data_cluster(device):
    """
        init SFT_Data_Cluster
        load center points in certain sft dataset
    """
    sft_data_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/multimodal-sft/llava_150k/en/llava_conver_single_turn_257k_clean.json"
    experiment_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/kmeans/{}/".format(sft_data_path.split('/')[-1].split('.')[0])
    sdc = SFT_Data_Cluster(sft_data_path, experiment_path, device)
    sdc.init_model() # load blip2 pretrain stage1
    sdc.load_center_points()
    return sdc

def split_samples_by_sim(test_split, dataloader, sdc, split_save_path):
    samples_by_cluster = defaultdict(list)
    for sample in tqdm(dataloader, desc=test_split):
        text_embeds = sdc.calculate_embedding(sample)    
        cluster_ids_y = kmeans_predict(
                X=text_embeds, cluster_centers=sdc.dis_center_points, distance='euclidean', device=sdc.device
            )
        for question_id, image_path, text_input, text_output, full_answer,cluster_id in zip(sample['question_id'], sample['image_path'], sample['text_input'], sample['text_output'], sample['fullAnswer'],cluster_ids_y):
            cluster = cluster_id.item()
            tmp = {
                'question_id': str(question_id.item()),
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

    split_save_path = os.path.join(split_save_path, test_split+'.json')
    print("Saving {} split to {}".format( test_split, split_save_path))
    with open(split_save_path, "w") as f:
        f.write(json.dumps(samples_by_cluster))
    
    return {test_split : statistic}

def mme_split_by_cluster(device, split_save_path):

    # init sft data cluster
    sdc = init_sft_data_cluster(device)
    
    # generate mme split by cluster
    test_list = list(SPLIT_MAP.keys())
    statistics = list()
    for test_split in test_list:
        ann_path = SPLIT_MAP[test_split]
        ann_path = os.path.join(MME_ROOT, ann_path)
        dataset = MMERawDataset(vis_root=MME_ROOT,ann_path=ann_path)
        dataloader = create_dataloader(dataset, batch_size=4)
        statistic = split_samples_by_sim(test_split, dataloader, sdc, split_save_path)
        statistics.append(statistic)
        print(statistic)
    
    split_statistics_save_path = os.path.join(split_save_path, 'statistics.json')
    with open(split_statistics_save_path, "w") as f:
        f.write(json.dumps(statistics))

def gqa_split_by_cluster(device, split_save_path):
    # init sft data cluster
    sdc = init_sft_data_cluster(device)
    ann_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/testdev_balanced_questions.json"
    vis_root = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/images/"

    # generate mme split by cluster
    dataset = GQARawDataset(vis_root, ann_path)
    dataloader = create_dataloader(dataset, batch_size=4)
    statistics = split_samples_by_sim("gqa", dataloader, sdc, split_save_path)
    print(statistics)

    split_statistics_save_path = os.path.join(split_save_path, 'statistics.json')
    with open(split_statistics_save_path, "w") as f:
        f.write(json.dumps(statistics))

def main():
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    
    # init config
    # /mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/mme_zeroshot_flant5xxl_instructblip_eval.yaml
    # cfg = Config(parse_args())
    # setup_seeds(cfg)
    # split_save_path = cfg.config.run.split_save_path
    # mme_split_by_cluster(device, split_save_path)

    split_save_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/kmeans_split/GQA/split_sim_0910/"
    if not os.path.exists(split_save_path):
        os.mkdir(split_save_path)
    gqa_split_by_cluster(device, split_save_path)


if __name__ == '__main__':
    main()