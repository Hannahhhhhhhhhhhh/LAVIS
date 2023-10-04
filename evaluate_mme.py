# import argparse

# import torch
# import lavis.tasks as tasks
import argparse
import random
import os
import numpy as np
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

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

# torch.cuda.set_device("cuda:6")

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
        default="lavis/projects/blip2/eval/mindgpt_visual.yaml",
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

class MMEDataset(Dataset):
    def __init__(self, vis_processor=None, vis_root=None, ann_path=None):
        self.vis_root = vis_root
        self.tmp_annotation = json.load(open(ann_path, 'r'))
        self.vis_processor = vis_processor
        self.annotation = list()

        self.questions, self.labels = [], []
        for ann in self.tmp_annotation:
            for one_question in ann['questions']:
                question, label = one_question.split('\t')
                self.annotation.append({
                    'image':ann['image'],
                    'question':question
                })
                self.questions.append(question)
                self.labels.append(label)
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        # question, label = ann['questions'].split('\t')
        question = self.questions[index]
        label = self.labels[index]

        image_path = '/'.join(ann['image'].split('/')[-2:])
        image = Image.open(os.path.join(self.vis_root, image_path)).convert("RGB")
        image = self.vis_processor(image)

        return {
            "image_name": image_path.split('/')[-1],
            "image": image,
            "text_input": question.strip(),
            "text_output": label.strip()
        }      

def create_dataloader(ann_path, vis_processors, cfg):
    dataset = MMEDataset(
        vis_processor=vis_processors['eval'],
        vis_root="/mnt/pfs-guan-ssai/nlu/dingyifeng/multimodal/MME_Benchmark_release",
        ann_path=ann_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.config.run.batch_size_eval,
        num_workers=cfg.config.run.num_workers,
        shuffle=False,
        drop_last=False
    )

    return dataloader


@torch.no_grad()
def evaluate_split(model, cfg, dataloader, output_path, test_split, device):
    image_names = []
    questions = []
    labels = []
    predictions = []
    t = 0
    for sample in tqdm(dataloader, desc=test_split):
        sample['image'] = sample['image'].to(device)
        sample['t'] = t
        if t == 0:
            print(sample['text_input'])
            print(sample['text_output'])

        outputs = model.predict_answers(
            samples=sample,
            inference_method=cfg.config.run.inference_method,
            num_beams=cfg.config.run.num_beams,
            max_len=cfg.config.run.max_len,
            min_len=cfg.config.run.min_len,
            prompt=cfg.config.run.prompt,
        )

        if t == 0:
            print(outputs)
        t += 1
        processed_outputs = [output.replace('\n', '_') for output in outputs]
        image_names += sample['image_name']
        questions += sample['text_input']
        labels += sample['text_output']
        predictions += processed_outputs

    output_data = list()
    with open(output_path, 'w') as f:
        for image_name, question, label, pred in zip(image_names, questions, labels, predictions):
            judge = 1 if label.lower()==pred.lower() else 0
            tmp = {
                'image':image_name,
                'question':question,
                'label':label.lower(),
                'pred':pred.lower(),
                'judge':judge
            }
            output_data.append(tmp)
            f.write(f"{json.dumps(tmp)}\n")
    print(f"save results to {output_path}")

    df = pd.DataFrame(output_data)
    df.to_csv(output_path.replace('json','csv'))


def analyze_mme_result(saved_path, mme_test_list, result_file):

    df_dict, result_dict = dict(), dict()
    for mme_test in mme_test_list:
        data_lst = list()
        with open(os.path.join(saved_path,f'{mme_test}.json'),"r") as fin:
            for line in fin:
                data = eval(line.strip())
                data_lst.append(data)
        df = pd.DataFrame(data_lst)

        # df_tmp = df.groupby('image').sum('judge')
        # df_acc = df_tmp[df_tmp['judge']==2]
        # acc_plus = df_acc.shape[0]/ df_tmp.shape[0] * 100
        # 有时候pred是一句话，取第一个单词与label判断
        df['new_judge'] = [1 if df['label'][i][0]==df['pred'][i][0] else 0 for i in range(df.shape[0])] 

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
    df_final.to_csv(result_file)

    perception_score = df_final['Accuracy_final'][:10].sum()
    cognition_score = df_final['Accuracy_final'][10:].sum()
    print(f"Perception Score: {perception_score}; Cognition Score: {cognition_score}")

def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # init
    cfg = Config(parse_args())
    setup_seeds(cfg)

    if not os.path.exists(cfg.config.run.output_dir):
        os.makedirs(cfg.config.run.output_dir)
    # assert len(os.listdir(cfg.config.run.output_dir)) == 0

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg).to(device)

    model.eval()

    _cfg = OmegaConf.load(cfg.config.model.preprocess_config)
    preprocess_cfg = _cfg.preprocess
    vis_processors, txt_processors = load_preprocess(preprocess_cfg)


    # test one piece
    raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})
    print(caption)

    # test_list = cfg.config.run.test_list
    test_list = list(SPLIT_MAP.keys())

    for test_split in test_list:
        print(f"Evaluating {test_split}")
        output_path = os.path.join(cfg.config.run.output_dir, test_split+'.json')
        if os.path.exists(output_path):
            continue

        ann_path = SPLIT_MAP[test_split]
        ann_path = os.path.join(MME_ROOT, ann_path)
        dataloader = create_dataloader(ann_path, vis_processors, cfg)

        evaluate_split(model, cfg, dataloader, output_path, test_split, device)

    # analyze the result of mme
    result_file = cfg.config.run.output_dir + cfg.config.run.result_output
    analyze_mme_result(cfg.config.run.output_dir, test_list, result_file)

if __name__ == '__main__':
    main()

    # lavis/projects/blip2/eval/mme_zeroshot_flant5xxl_eval.yaml
    # lavis/projects/blip2/eval/gqa_sft_eval/mme_gqa_sft_flant5xxl_eval_0924.yaml

