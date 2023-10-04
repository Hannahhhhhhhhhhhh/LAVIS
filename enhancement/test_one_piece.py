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

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from lavis.models import load_preprocess
from lavis.common.vqa_tools.vqa_eval import VQAEval

import sys
sys.path.append("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS")
from enhancement.k_means import SFT_Data_Cluster
from kmeans_pytorch import kmeans, kmeans_predict

from collections import defaultdict


SFT_PATH = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/sft/llava_st_257k_clean_k_means_5_train_cluster_0915/"
CLUSTER_CKPT = {
    'cluster_0': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster0/20230918192/checkpoint_2.pth",
    'cluster_1': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster1/20230915211/checkpoint_2.pth",
    'cluster_2': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster2/20230915193/checkpoint_2.pth",
    'cluster_3': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster3/20230915213/checkpoint_2.pth",
    'cluster_4': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster4/20230915220/checkpoint_2.pth",
    'cluster_5': "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/blip2/blip2-flant5-xxl/blip2_pretrained_flant5xxl.pth",
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
    # parser.add_argument("-f", help="jupyter notebook")
    parser.add_argument("--cfg-path",
        required=False,
        help="path to configuration file.",
        # default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/mme_zeroshot_flant5xxl_instructblip_eval.yaml",
        default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/cluster_eval/gqa_zeroshot_flant5xxl_sft_eval_default.yaml",
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

        
def test_one_piece(device, vis_processors, model, sample, cfg):
    raw_image = Image.open(sample['image_path']).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    sample['image'] = image
    outputs = model.predict_answers(
        samples=sample,
        inference_method=cfg.config.run.inference_method,
        num_beams=cfg.config.run.num_beams,
        max_len=cfg.config.run.max_len,
        min_len=cfg.config.run.min_len,
        prompt=cfg.config.run.prompt,
    )
    return outputs[0]

def main():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    # init config
    cfg = Config(parse_args())
    setup_seeds(cfg)

    # init vis_processors, txt_processors
    _cfg = OmegaConf.load(cfg.config.model.preprocess_config)
    preprocess_cfg = _cfg.preprocess
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg).to(device)
    model.eval()
    print(model.query_tokens)


    gqa_data = json.load(open(cfg.config.run.split_save_path + 'gqa.json',"r"))

    gqa_total_data = list()
    for k,v in gqa_data.items():
        gqa_total_data += v

    predict_all = list()
    for sample in tqdm(gqa_total_data):
        output_lst = list()
        tmp = sample.copy()
        for cluster_id in range(6):
            ckpt_path = CLUSTER_CKPT.get("cluster_{}".format(cluster_id))
            tt = model.load_checkpoint(url_or_filename=ckpt_path)
            # print(model.query_tokens)
            output = test_one_piece(device, vis_processors, model, sample, cfg)
            output_lst.append(output)
            tmp[f"pred_c{cluster_id}"] = output
            tmp[f"acc_c{cluster_id}"] = 1 if sample['text_output'] in output else 0
        tmp['final'] = 1 if sample['text_output'] in output_lst else 0
        tmp['image'] = ""
        predict_all.append(tmp)

    df = pd.DataFrame(predict_all)
    df.to_csv("/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/enhancement/test_0920.csv",index=False)


if __name__ == "__main__":
    main()