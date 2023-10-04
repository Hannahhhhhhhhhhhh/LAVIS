import abc
import numpy as np
import os
import json
import torch
import random
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

# https://github.com/google/active-learning/blob/efedd8f1c45421ee13af2b9ff593ad31f3835942/sampling_methods/kcenter_greedy.py

class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None

class kCenterGreedy(SamplingMethod):

    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.

        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
            min_distances.
        rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                                if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric) # 所有点与cluster_centers之间的距离

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist) # min_distances.shape = (num_samples,1) 与每个cluster_centers距离的最小值

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
        already_selected: index of datapoints already selected
        N: batch size

        Returns:
        indices of points selected to minimize distance to cluster centers
        """

        try:
            self.features = self.X
            print('Calculating distances...')
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            print('Using flat_X as features.')
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances) # 取最小距离中最大的那个sample点，作为新的cluster_center
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'% max(self.min_distances))


        self.already_selected = already_selected

        return new_batch

def load_cluster_data_lst_by_cluster(sample_split_save_path, length=1000, num_cluster=5):
    cluster_data_lst = list()
    for cluster_id in range(num_cluster):
        split_sample = os.path.join(sample_split_save_path, f"{cluster_id}_cluster.json")
        split_data = json.load(open(split_sample,"r"))
        cluster_data_lst += split_data[:length]
    return cluster_data_lst

def load_cluster_data_lst(sft_data_path, length=1000):
    cluster_data_lst = list()
    data = json.load(open(sft_data_path,"r"))
    cluster_data_lst = random.sample(data, length)
    return cluster_data_lst

def load_embeddings(sample_embeds_save_path, cluster_data_lst):
    X = list()
    y = list()
    for cluster_data in tqdm(cluster_data_lst):
        file = cluster_data['id'] + '.pth'
        embed = torch.load(os.path.join(sample_embeds_save_path, file))
        X.append(embed.detach().numpy())
        try:
            y.append(cluster_data['cluster_id'])
        except:
            continue
    return X, y

def main():
    # lrcinstruction
    # sft_data_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/multimodal-sft/LrvInstruction/processed/lrvinstruction_152k_filtered_shorter_150_ids.json"
    # sample_embeds_save_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/cluster/{}/sample_embeds".format(sft_data_path.split('/')[-1].split('.')[0])
    # sample_split_save_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/cluster/{}/agnes/split_samples".format(sft_data_path.split('/')[-1].split('.')[0])
    # cluster_data_lst = load_cluster_data_lst_by_cluster(sample_split_save_path, length=100, num_cluster=5)

    # llava single turn 257k
    sft_data_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/data/multimodal-sft/llava_150k/en/llava_conver_single_turn_257k_clean_v2.json"
    sample_embeds_save_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/experiments/cluster/{}/sample_embeds".format(sft_data_path.split('/')[-1].split('.')[0])
    cluster_data_lst = load_cluster_data_lst(sft_data_path, length=1000)

    X, y = load_embeddings(sample_embeds_save_path, cluster_data_lst)

    kcg = kCenterGreedy(np.array(X), y, seed=2023)

    new_batch = kcg.select_batch_(already_selected=[], N=20)

    print(new_batch)
    for sample_id in new_batch:
        print(cluster_data_lst[sample_id]['text_input'])


if __name__ == '__main__':
    main()