import json
import os 
import sys
from tqdm import tqdm
from collections import defaultdict
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# sys.path.append("/workspace/code/LAVIS")
from lavis.models import load_model

from kmeans_pytorch import kmeans, kmeans_predict


class SFTDataset(Dataset):
    def __init__(self, sft_data_path):
        self.annotation = json.load(open(sft_data_path, "r"))
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        return self.annotation[index]

def create_dataloader(sft_data_path, batch_size):
    dataset = SFTDataset(sft_data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    return dataset, dataloader

class SFT_Data_Cluster():
    def __init__(self, cfg):
        self.cfg = cfg
        self.sft_data_path = cfg.dataset.sft_data_path
        self.device = torch.device(f"cuda:{cfg.run.device}" if torch.cuda.is_available() else "cpu")
        
        self.samples = list()
        self.sample_dataloader = None
        self.sample_embeddings =  dict()
        self.experiment_path = os.path.join(cfg.run.experiment_raw_path, self.sft_data_path.split('/')[-1].split('.')[0])
        self.sample_embeds_save_path = os.path.join(self.experiment_path,'sample_embeds')

        self.model = None
        self.samples_to_cluster = dict() # {'sample_id':'cluster_id',...}
        self.cluter_to_samples = defaultdict(list) # {'cluster_id':[sample_id, ...],...}
        self.cluster_specific_path = os.path.join(self.experiment_path, cfg.run.type)
        self.split_samples_path = os.path.join(self.cluster_specific_path, 'split_samples')
        self.cluter_to_samples_file = os.path.join(self.cluster_specific_path,'cluter_to_samples.json')
        self.samples_to_cluster_file = os.path.join(self.cluster_specific_path,'samples_to_cluster.json')
        self.dis_center_points = None
        self.dis_center_points_save_path = os.path.join(self.cluster_specific_path,'distribution_center_points.pth')
        self.task_center_points = dict()

        self.init_paths()
        self.print_key_config()


    def load_data(self):
        self.samples, self.sample_dataloader = create_dataloader(self.sft_data_path, batch_size=self.cfg.dataset.batch_size)
        print(f"Loading {len(self.samples)} samples from path: ", self.sft_data_path)

    def init_paths(self):
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)
        if not os.path.exists(self.cluster_specific_path):
            os.mkdir(self.cluster_specific_path)
        if not os.path.exists(self.sample_embeds_save_path):
            os.mkdir(self.sample_embeds_save_path)
        if not os.path.exists(self.split_samples_path):
            os.mkdir(self.split_samples_path)

    def init_model(self):
        print(f"Loading encoding model to device {self.device} ...,\nmodel arch: {self.cfg.model.arch}, model type: {self.cfg.model.model_type}")
        self.model = load_model(self.cfg.model.arch, self.cfg.model.model_type, is_eval=True, device=self.device) # BLIP2 first-stage model with Q-former and ViT.

    def print_key_config(self):
        print(" ------------- CONFIG ------------- ")
        print("sft_data_path: ", self.sft_data_path)
        print("device: ", self.device)
        print("experiment path: ", self.experiment_path)
        print("sample embedding save path: ", self.sample_embeds_save_path)
        print("split samples path: ", self.split_samples_path)

    def calculate_embedding(self, sample):
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
        # Reference:
        # encode all samples using a pre-trained language model
        # 1. extract the last_hidden_state of each sample
        # 2. mean pooling on the word embeddings of each sample
        # 3. obtain one-dimentional vector as the sentence embedding
        # 4. L2 normalization on the embedding dimension

        # Done:
        # 1. use blip2 first-stage model to encode text(question)
        # 2. sample embeddings saved to self.sample_embeds_save_path

        if self.cfg.run.re_encode:
            self.init_model()
       
            for sample in tqdm(self.sample_dataloader, desc="Encode Samples"):
                text_embeds = self.calculate_embedding(sample)
                sample_ids = sample['id']
                for sample_id, text_embed in zip(sample_ids, text_embeds):
                    torch.save(text_embed.to(torch.device('cpu')), os.path.join(self.sample_embeds_save_path,f"{sample_id}.pth"))

        print("Loading sample embeddings from : ",self.sample_embeds_save_path)
        for file in tqdm(os.listdir(self.sample_embeds_save_path)):
            y = torch.load(os.path.join(self.sample_embeds_save_path, file))
            self.sample_embeddings[file.split('.')[0]] = y
        print(self.sample_embeddings[file.split('.')[0]].shape)

        assert len(self.samples) == len(self.sample_embeddings), f"Length of sample_embeddings({len(self.sample_embeddings)}) should be the same as samples({len(self.samples)})."

    def k_means(self):
        # unsupervised clustering in the embedding space
        # obtain the mapping of each sample and its corresponding cluster label
        # select the center point of the cluster as distribution center point of that downstream task
        # reference: https://blog.csdn.net/weixin_52456426/article/details/127086334
        # other: https://zhuanlan.zhihu.com/p/65331686?utm_id=0

        tmp_data = [value.unsqueeze(dim=0).detach() for value in list(self.sample_embeddings.values())]
        data = torch.cat(tmp_data)
        # print：数据集数量，数据集维数，聚类的类别数
        print(f"data size:{data.shape}, num_clusters:{self.cfg.run.num_clusters}")

        # 训练阶段
        # X：待聚类数据集(需要是torch.Tensor类型)，维数，距离计算法则，训练设备
        cluster_ids_x, cluster_centers = kmeans(
            X=data, num_clusters=self.cfg.run.num_clusters, distance=self.cfg.run.distance, device=self.device
        )

        # save cluster_centers
        self.dis_center_points = cluster_centers

        # save sample_ids in each cluster
        for i in tqdm(range(len(cluster_ids_x)), desc="Sample Label"):
            cluster_id = cluster_ids_x[i].item()
            self.cluter_to_samples[cluster_id].append(self.samples[i]['id'])
            self.samples_to_cluster[self.samples[i]['id']] = cluster_id
    
    def cluster_sample_save_statistic(self):
        print(f"Save cluster samples to: \n{self.cluter_to_samples_file} \n{self.samples_to_cluster_file}")
        for k, v in self.cluter_to_samples.items():
            print(f"Cluster {k} having {len(v)} samples;")
        with open(self.cluter_to_samples_file, "w") as f:
            f.write(f"{json.dumps(self.cluter_to_samples)}")
        with open(self.samples_to_cluster_file, "w") as f:
            f.write(f"{json.dumps(self.samples_to_cluster)}")

        print("Save distribution center points to:", self.dis_center_points_save_path)
        torch.save(self.dis_center_points.to(torch.device('cpu')), self.dis_center_points_save_path)

    def split_sft_data_by_cluster(self):
        # TODO
        if len(self.samples_to_cluster) == 0:
            self.samples_to_cluster = json.load(open(self.samples_to_cluster_file, "r"))
        tmp_split = defaultdict(list)
        for sample in tqdm(self.samples):
            cluster_id = self.samples_to_cluster[sample['id']]
            tmp = sample.copy()
            tmp['cluster_id'] = cluster_id
            tmp_split[cluster_id].append(tmp)
        for k,v in tmp_split.items():
            with open(os.path.join(self.split_samples_path, f"{str(k)}_cluster.json")):
                f.write(v)

    def load_center_points(self):
        self.dis_center_points = torch.load(self.dis_center_points_save_path)

    def predict_kmeans(self, test):

        cluster_ids_y = kmeans_predict(
            X=test, cluster_centers=self.dis_center_points, distance=self.cfg.run.distance, device=self.device
        )
        print(cluster_ids_y)

    def cosing_similarity(embed1, embed2):

        return similarity

    def select_task_center_points(dis_center_points, sample_embeddings):
        # task center point is one exact sample from this task data with the biggest cosine similarity to the distribution center point
        """
            return: task_center_points = {cluster_id: center_point_id, ... }
        """
        return task_center_points

    def clustering(sample_embeddings):
        # do k_means, give each sample an label, get the distribution center points for each cluster
        cluster_label, dis_center_points = k_means(sample_embeddings)

        # select the nearest exact sample to the distribution center points as the task center points
        task_center_points = select_task_center_points(dis_center_points, sample_embeddings)

        return cluster_label, dis_center_points, task_center_points

    def coreset_sampling():
        # coreset algorithm KCentergreedy, select set of core samples from the task samples
        # choose k center points such that minimize the largest distance between a random data point and its nearest center,
        return 0

    def train(self):
        ## train
        # sentence embedding
        self.encode()
        # # clustering
        self.k_means()
        self.cluster_sample_save_statistic()
        # coreset sampling, 从每个cluster中选出代表性的samples(Optimal)
        # coreset_sampling()
        self.split_sft_data_by_cluster()
    
    def predict(self, test_sample):
        ## predict
        self.load_center_points()
        self.predict_kmeans(test_sample)

def main():
    config_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/enhancement/config/llava_single_turn_257k_cluster.yaml"
    # config_file = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/LAVIS/enhancement/config/lrvinstruction_152k_filtered_shorter_150_cluster.yaml"
    cfg = OmegaConf.load(config_file)

    sdc = SFT_Data_Cluster(cfg)
    # load input sft data
    sdc.load_data()

    sdc.encode()

    # if cfg.run.train:
        # sdc.train()

    # if cfg.run.predict:
        # test_sample = torch.rand(5,256)
        # sdc.predict(test_sample)


if __name__ == "__main__":
    main()