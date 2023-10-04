import json
import os 
import sys
import time
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
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import metrics

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

class SFT_Data_Clu

class SFT_Data_Cluster():
    def __init__(self, cfg):
        self.cfg = cfg
        self.sft_data_path = cfg.dataset.sft_data_path
        self.device = torch.device(f"cuda:{cfg.run.device}" if torch.cuda.is_available() else "cpu")
        
        self.samples = list()
        self.sample_dataloader = None
        self.sample_ids = list()
        self.sample_embeddings =  list()
        self.experiment_path = os.path.join(cfg.run.experiment_raw_path, self.sft_data_path.split('/')[-1].split('.')[0])
        self.sample_embeds_save_path = os.path.join(self.experiment_path,'sample_embeds')

        self.model = None
        self.samples_to_cluster = dict() # {'sample_id':'cluster_id',...}
        self.cluster_specific_path = os.path.join(self.experiment_path, cfg.run.type)
        self.split_samples_path = os.path.join(self.cluster_specific_path, 'split_samples')
        self.samples_to_cluster_file = os.path.join(self.cluster_specific_path,'samples_to_cluster.json')
        self.dis_center_points = None
        self.dis_center_points_save_path = os.path.join(self.cluster_specific_path,'distribution_center_points.pth')

        self.init_paths()
        self.print_key_config()


    def load_data(self):
        self.samples, self.sample_dataloader = create_dataloader(self.sft_data_path, batch_size=self.cfg.dataset.batch_size)
        self.sample_ids = [ sample['id'] for sample in self.samples]
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
        print("cluster specific path: ", self.cluster_specific_path)
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
        for sample_id in tqdm(self.sample_ids, desc="load embedding"):
            file_path = os.path.join(self.sample_embeds_save_path, f"{sample_id}.pth")
            embed = torch.load(file_path)
            self.sample_embeddings.append(embed.detach().numpy())

        print(self.sample_embeddings[0].shape)

        # assert len(self.samples) == len(self.sample_embeddings), f"Length of sample_embeddings({len(self.sample_embeddings)}) should be the same as samples({len(self.samples)})."

    # def k_means(self, data):
    #     # k-means
    #     # select the center point of the cluster as distribution center point of that downstream task
    #     # reference: https://blog.csdn.net/weixin_52456426/article/details/127086334
    #     # other: https://zhuanlan.zhihu.com/p/65331686?utm_id=0

    #     cluster_ids_x, cluster_centers = kmeans(
    #         X=data, 
    #         num_clusters=self.cfg.run.num_clusters, 
    #         distance=self.cfg.run.distance, 
    #         device=self.device
    #     )
    #     # save cluster_centers
    #     self.dis_center_points = cluster_centers
    #     return cluster_ids_x
    
    def k_means(self, data):
        # k-means realized by sklearn

        kmeans = KMeans(
            n_clusters=self.cfg.run.num_clusters,
            random_state=0, 
            n_init="auto"
        ).fit(data)
        y = kmeans.labels_
        self.dis_center_points = torch.Tensor(kmeans.cluster_centers_)
        return y


    def agnes_cluster(self, data):
        # 层次聚类 AGNES，Agglomerative Nesting
        # should set distance_threshold or n_clusters, and set another one to be None
        # distance_threshold(float): The linkage distance threshold at or above which clusters will not be merged.
        # n_clusters(int): The number of clusters to find. 
        # reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
        start = time.time()
        clustering = AgglomerativeClustering(
            linkage = self.cfg.run.linkage, 
            distance_threshold=self.cfg.run.get('distance_threshold',None), 
            n_clusters=self.cfg.run.get('n_clusters',None)
        ).fit(data)
        end = time.time()
        print(f"AgglomerativeClustering cost {(end-start)}s")
        print("Agglomerative cluster number: ", clustering.n_clusters_)
        y = clustering.labels_
        return y

    def cluster(self):
        # unsupervised clustering in the embedding space
        # obtain the mapping of each sample and its corresponding cluster label

        data = self.sample_embeddings
        print(f"data size:{len(data)}")

        if self.cfg.run.type == 'kmeans':
            cluster_ids_y = self.k_means(data)
        elif self.cfg.run.type == 'agnes':
            cluster_ids_y = self.agnes_cluster(data)

        # metrics
        # if self.cfg.run.metrics == 'silhouette_score':
            # score = metrics.silhouette_score(data, cluster_ids_y, metric='euclidean')
            # print("Metrics silhouette score: ", score)

        # save sample_ids in each cluster
        for i in tqdm(range(len(cluster_ids_y)), desc="Sample Label"):
            self.samples_to_cluster[self.sample_ids[i]] =  str(cluster_ids_y[i])
        print(self.samples_to_cluster)

    def cluster_sample_save_statistic(self):
        print(f"Save cluster samples to: \n{self.samples_to_cluster_file}")
        with open(self.samples_to_cluster_file, "w") as f:
            f.write(f"{json.dumps(self.samples_to_cluster)}")

        if self.dis_center_points is not None:
            print("Save distribution center points to:", self.dis_center_points_save_path)
            torch.save(self.dis_center_points.to(torch.device('cpu')), self.dis_center_points_save_path)

    def split_sft_data_by_cluster(self):
        # split sft data by cluster
        if len(self.samples_to_cluster) == 0:
            self.samples_to_cluster = json.load(open(self.samples_to_cluster_file, "r"))

        tmp_split = defaultdict(list)
        for sample in tqdm(self.samples, desc="split_cluster"):
            try:
                cluster_id = self.samples_to_cluster[sample['id']]
                tmp = sample.copy()
                tmp['cluster_id'] = cluster_id
                tmp_split[cluster_id].append(tmp)
            except:
                continue
        
        for k,v in tmp_split.items():
            print(f"Cluster {k} has {len(v)} samples...")
            with open(os.path.join(self.split_samples_path, f"{str(k)}_cluster.json"),"w") as f:
                f.write(json.dumps(v))

    def coreset_sampling():
        # coreset algorithm KCentergreedy, select set of core samples from the task samples
        # choose k center points such that minimize the largest distance between a random data point and its nearest center,
        return 0

    def train(self):
        ## train
        # sentence embedding
        self.encode()
        
        # # clustering
        self.cluster()
        self.cluster_sample_save_statistic()
        
        # split dataset by cluster
        self.split_sft_data_by_cluster()

        # coreset sampling, 从每个cluster中选出代表性的samples(Optimal)
        # coreset_sampling()
    
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

    if cfg.run.train:
        sdc.train()

    # if cfg.run.predict:
        # test_sample = torch.rand(5,256)
        # sdc.predict(test_sample)


if __name__ == "__main__":
    main()