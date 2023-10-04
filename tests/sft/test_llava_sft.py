import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from omegaconf import OmegaConf

import torch
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis")

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import LayerNorm
import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank
from lavis.models import load_preprocess

from torch.utils.data import DataLoader
from lavis.datasets.datasets.instruction_datasets import LlavaSftInstructionDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    # parser.add_argument("-f", help="jupyter notebook")
    parser.add_argument("--cfg-path", default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/train/llava_sft_single_turn_total_0904.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def create_dataloader(cfg, vis_processors,txt_processors, batch_size=32):
    ann_paths = [cfg.config.datasets.llava_150k_en_sft_singleturn.build_info.annotations.train.storage]
    vis_root = cfg.config.datasets.llava_150k_en_sft_singleturn.build_info.images.storage
    
    print("ann_paths:", ann_paths)
    print("vis_root:", vis_root)
    dataset = LlavaSftInstructionDataset(
        vis_processor=vis_processors['eval'],
        text_processor=txt_processors["eval"],
        vis_root=vis_root,
        ann_paths=ann_paths
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataset, data_loader

def forward(dataloader, device, model):
    pred_qa_pairs = list()
    cnt = 1
    for batch in tqdm(dataloader):
        batch['image'] = batch['image'].half().to(device)
        # batch['image'] = torch.rand(batch['image'].shape).half().to(device)
        batch['text_inputs'] = [text for text in batch['text_input']]        
        predict = model.predict_answers(batch)[0]
        print(predict)
        print(batch['text_input'], batch['text_output'])
        break

def main():
    # build config
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    cfg = Config(parse_args())
    setup_seeds(cfg)
    print(cfg._convert_node_to_json(cfg.config))

    # load vis_processor & text_processor & datasets
    preprocess_cfg = cfg.config.preprocess
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    dataset, dataloader = create_dataloader(cfg, vis_processors,txt_processors, batch_size=1)

    # load model & setup task
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg).to(device)
    model.eval()

    # forward
    forward(dataloader, device, model)

if __name__ == '__main__':
    main()
