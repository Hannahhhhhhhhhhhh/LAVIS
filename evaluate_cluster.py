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
# version 1
# SFT_PATH = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/sft/"
# CLUSTER_CKPT = {
#     'cluster_0':SFT_PATH + 'llava_single_turn_257k_sft_freeze_qf_train_qt_textinqf_epo3_0906_cluster0/20230906203/checkpoint_2.pth',
#     'cluster_1':SFT_PATH + 'llava_single_turn_257k_sft_freeze_qf_train_qt_textinqf_epo3_0906_cluster1/20230906231/checkpoint_2.pth',
#     'cluster_2':SFT_PATH + 'llava_single_turn_257k_sft_freeze_qf_train_qt_textinqf_epo3_0907_cluster2/20230907112/checkpoint_2.pth',
#     'cluster_3':SFT_PATH + 'llava_single_turn_257k_sft_freeze_qf_train_qt_textinqf_epo3_0907_cluster3/20230907135/checkpoint_2.pth',
#     'cluster_4':SFT_PATH + 'llava_single_turn_257k_sft_freeze_qf_train_qt_textinqf_epo3_0907_cluster4/20230907150/checkpoint_2.pth'
# }

CLUSTER_NUM = 5

# version 2
SFT_PATH = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/blip2/flant5xxl/sft/llava_st_257k_clean_k_means_5_train_cluster_0915/"
CLUSTER_CKPT = {
    'cluster_0': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster0/20230918192/checkpoint_2.pth",
    'cluster_1': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster1/20230915211/checkpoint_2.pth",
    'cluster_2': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster2/20230915193/checkpoint_2.pth",
    'cluster_3': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster3/20230915213/checkpoint_2.pth",
    'cluster_4': SFT_PATH + "freeze_qf_train_qt_textinqf_epo3_cluster4/20230915220/checkpoint_2.pth",
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
        # default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/mme_zeroshot_flant5xxl_instructblip_eval.yaml",
        default="/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/gqa_zeroshot_flant5xxl_sft_eval_default.yaml",
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

def test_one_piece(device, vis_processors, model):
    raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})
    print(caption)

class ClusterDataset(Dataset):
    def __init__(self, vis_processor=None, ann_path=None, cluster=0):
        self.tmp_annotation = json.load(open(ann_path, 'r'))
        self.vis_processor = vis_processor
        if cluster in self.tmp_annotation.keys():
            self.annotation = self.tmp_annotation[cluster]
        elif str(cluster) in self.tmp_annotation.keys():
            self.annotation = self.tmp_annotation[str(cluster)]
        else:
            # print('Cluster {} not in dataset:{}'.format(cluster, ann_path))
            self.annotation = None
    
    def __len__(self):
        if self.annotation == None:
            return 0
        else:
            return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        image = Image.open(ann['image_path']).convert("RGB")
        image = self.vis_processor(image)

        return {
            "question_id": ann.get("question_id",""),
            "image_name": ann['image_path'].split('/')[-1],
            "image": image,
            "text_input": ann['text_input'].strip(),
            "text_output": ann['text_output'].strip(),
            "full_answer": ann.get("full_answer",""),
        }

def create_cluster_dataloader(ann_path, cluster_id, vis_processors, cfg):
    dataset = ClusterDataset(
        vis_processor=vis_processors['eval'],
        ann_path=ann_path,
        cluster=cluster_id
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.config.run.batch_size_eval,
        num_workers=cfg.config.run.num_workers,
        shuffle=False,
        drop_last=False
    )
    return dataset, dataloader

@torch.no_grad()
def evaluate_split(model, cfg, dataloader, output_path, test_split, device):
    output_data = []

    for sample in tqdm(dataloader, desc=test_split):
        sample['image'] = sample['image'].to(device)

        outputs = model.predict_answers(
            samples=sample,
            inference_method=cfg.config.run.inference_method,
            num_beams=cfg.config.run.num_beams,
            max_len=cfg.config.run.max_len,
            min_len=cfg.config.run.min_len,
            prompt=cfg.config.run.prompt,
        )

        processed_outputs = [output.replace('\n', '_') for output in outputs]

        for image_name, question, label, question_id, full_answer, pred in zip(sample['image_name'], sample['text_input'], sample['text_output'], sample['question_id'], sample['full_answer'], processed_outputs):
            judge = 1 if label.lower() == pred.lower() else 0
            tmp = {
                'question_id': question_id,
                'image': image_name,
                'question': question,
                'gt_ans': label.lower(),
                'pred_ans': pred.lower(),
                'full_answer': full_answer,
                'judge': judge
            }
            output_data.append(tmp)

    with open(output_path, 'w') as f:
        for data in output_data:
            f.write(f"{json.dumps(data)}\n")

    print(f"save results to {output_path}")

    df = pd.DataFrame(output_data)
    df.to_csv(output_path.replace('json','csv'))

def evaluate_mme_by_cluster(model, cfg, vis_processors, cluster_id, test_list, device):
    cluster_output = os.path.join(cfg.config.run.output_dir, cluster_id)
    if not os.path.exists(cluster_output):
        os.mkdir(cluster_output)

    for test_split in test_list:
        print(f"Evaluating {test_split}")
        output_path = os.path.join(cluster_output, test_split+'.json')
        if os.path.exists(output_path):
            continue

        ann_path = os.path.join(cfg.config.run.split_save_path, SPLIT_MAP[test_split])
        dataset, dataloader = create_cluster_dataloader(ann_path, cluster_id, vis_processors, cfg)
        print("MME Cluster Evaluation: Cluster {} having {} samples...".format(cluster_id, len(dataset)))
        if len(dataset) == 0:
            continue
        evaluate_split(model, cfg, dataloader, output_path, test_split, device)

def evaluate_gqa_by_cluster(model, cfg, vis_processors, cluster_id, device):
    cluster_output = os.path.join(cfg.config.run.output_dir, cluster_id)
    if not os.path.exists(cluster_output):
        os.mkdir(cluster_output)
    
    output_path = os.path.join(cluster_output, 'gqa.json')
    ann_path = os.path.join(cfg.config.run.split_save_path,'gqa.json')
    dataset, dataloader = create_cluster_dataloader(ann_path, cluster_id, vis_processors, cfg)
    print("GQA Cluster Evaluation: Cluster {} having {} samples...".format(cluster_id, len(dataset)))
    if len(dataset) == 0:
        print("error")
        return 0
    evaluate_split(model, cfg, dataloader, output_path, 'gqa', device)
    
def evaluate_by_cluster(model, cfg, vis_processors, evaluation_type='mme', device='cuda'):
    if not os.path.exists(cfg.config.run.output_dir):
        os.makedirs(cfg.config.run.output_dir)
    
    # evaluate by clusters
    for cluster_id in range(CLUSTER_NUM):
        print("Inference {} for Cluster {}".format(evaluation_type, cluster_id))
        ckpt_path = CLUSTER_CKPT.get("cluster_{}".format(cluster_id))
        model.load_checkpoint(url_or_filename=ckpt_path)
        print(model.query_tokens)
        if evaluation_type == 'mme':
            test_list = SPLIT_MAP.keys()
            evaluate_mme_by_cluster(model, cfg, vis_processors, str(cluster_id), test_list, device)
        elif evaluation_type == 'gqa':
            evaluate_gqa_by_cluster(model, cfg, vis_processors, str(cluster_id), device)
            _ = analyze_gqa_one_cluster(cfg, cluster_id)
    
    # analyze evaluation result
    if evaluation_type == 'gqa':
        analyze_gqa_result(cfg)
    elif evaluation_type == 'mme':
        analyze_mme_result(cfg)
        
def analyze_mme_result(cfg):

    mme_test_list = SPLIT_MAP.keys()
    df_dict, result_dict = dict(), dict()
    for mme_test in mme_test_list:
        data_lst = list()
        for cluster_id in range(CLUSTER_NUM):
            saved_file = os.path.join(cfg.config.run.output_dir, str(cluster_id), f'{mme_test}.json')
            if os.path.exists(saved_file):
                with open(saved_file,"r") as fin:
                    for line in fin:
                        data = eval(line.strip())
                        data_lst.append(data)

        df = pd.DataFrame(data_lst)

        if len(data_lst) == 0:
            continue

        # df_tmp = df.groupby('image').sum('judge')
        # df_acc = df_tmp[df_tmp['judge']==2]
        # acc_plus = df_acc.shape[0]/ df_tmp.shape[0] * 100
        # 有时候pred是一句话，取第一个单词与label判断
        df['new_judge'] = [1 if df['gt_ans'][i].split(',')[0]==df['pred_ans'][i].split(',')[0] else 0 for i in range(df.shape[0])] 

        df_tmp = df.groupby('image').sum('new_judge')
        df_acc = df_tmp[df_tmp['new_judge']==2]
        acc_plus = df_acc.shape[0]/ df_tmp.shape[0] * 100
        acc = df['new_judge'].sum()/len(df['new_judge']) * 100

        result = {
            "Accurate" : df['new_judge'].sum(),
            "Accurate_both" : df_acc.shape[0],
            "Number" : len(df['new_judge']),
            "Accuracy" : acc,
            "Accuracy+" : acc_plus,
            "Accuracy_final" : acc +acc_plus
        }
        print(mme_test, " : ", result)
        df_dict[mme_test] = df
        result_dict[mme_test] = result

    df_output = pd.DataFrame(result_dict).T
    order = ['existence','count','position','color','OCR','posters','celebrity','scene','landmark','artwork','commonsense_reasoning','numerical_calculation','text_translation','code_reasoning']
    df_final = df_output.loc[order]
    result_file = os.path.join(cfg.config.run.output_dir, cfg.config.run.result_output)
    print("Result save to: ", result_file)
    df_final.to_csv(result_file)

    perception_score = df_final['Accuracy_final'][:10].sum()
    cognition_score = df_final['Accuracy_final'][10:].sum()
    print(f"Perception Score: {perception_score}; Cognition Score: {cognition_score}")

def cal_acc(data, result_file):
    acc = []
    vqa_tool = VQAEval()
    results = list()
    for i in range(len(data)):
        res = data[i]

        gt_ans = res["gt_ans"]
        pred = res["pred_ans"]

        pred = vqa_tool.processPunctuation(pred)
        pred = vqa_tool.processDigitArticle(pred)
        
        vqa_acc = 1 if gt_ans in pred else 0
        # vqa_acc = 1 if pred == gt_ans else 0
        
        acc.append(vqa_acc)
        res['vqa_acc']= vqa_acc
        results.append(res)

    accuracy = sum(acc) / len(acc) * 100
    metrics = {"agg_metrics": accuracy, "acc": accuracy}
    
    # new_metrics = vqa_answer_eval(result_file)
    # metrics.update(new_metrics)
        
    print(metrics)
    return results

def analyze_gqa_one_cluster(cfg, cluster_id):
    gqa_image_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/GQA/images/"

    data = list()
    gqa_cluster_result = os.path.join(cfg.config.run.output_dir, "{}/gqa.json".format(str(cluster_id)))
    with open(gqa_cluster_result,"r") as f:
        for line in f:
            tmp = json.loads(line)
            tmp['image_id'] = tmp['image']
            tmp['image'] = os.path.join(gqa_image_path, tmp['image_id'])
            data.append(tmp)
    print(cluster_id, " : ", len(data))
    cal_acc(data, None)
    return data

def analyze_gqa_result(cfg):
    # load gqa predict file(cluster)

    total_data = list()
    for cluster_id in range(CLUSTER_NUM):
        data = analyze_gqa_one_cluster(cfg, cluster_id)
        total_data += data

    with open(os.path.join(cfg.config.run.output_dir, "gqa_total.json"),"w") as f:
        f.write(json.dumps(total_data))
    print(total_data[0], len(total_data))
    cal_acc(total_data, None)


def main():
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    # init config
    cfg = Config(parse_args())
    setup_seeds(cfg)

    # init vis_processors, txt_processors
    _cfg = OmegaConf.load(cfg.config.model.preprocess_config)
    preprocess_cfg = _cfg.preprocess
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)

    # init task & model
    task = tasks.setup_task(cfg)
    model = task.build_model(cfg).to(device)
    model.eval()

    print(model.query_tokens)
    test one piece
    test_one_piece(device, vis_processors, model)
    # model = None

    # evaluate by clusters
    evaluation_type =  cfg.config.run.type # gqa or mme
    evaluate_by_cluster(model, cfg, vis_processors,evaluation_type, device)


if __name__ == '__main__':
    main()

    # python evaluate_cluster.py --cfg-path /mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/cluster_eval/gqa_zeroshot_flant5xxl_sft_eval_default.yaml
    # python evaluate_cluster.py --cfg-path /mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/lavis/projects/blip2/eval/cluster_eval/mme_zeroshot_flant5xxl_sft_eval_default.yaml
