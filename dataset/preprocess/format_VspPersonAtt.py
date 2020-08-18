import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

group_order = [i for i in range(56)]


def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    dataset = EasyDict()
    dataset.description = 'pa100k'
    dataset.reorder = 'group_order'
    dataset.root = '/mnt/data/user/zhangqinlu/dataset/person_dataset/new_CCF_PA100K/data/'
    part_pkl_path = '../partition_pa100_ccf100_56_0427.pkl'
    data_pkl_path = '../dataset_pa100_ccf100_56_0427.pkl'
    partition_vsp = pickle.load(open(part_pkl_path, 'rb'))
    dataset_vsp = pickle.load(open(data_pkl_path, 'rb'))
    dataset.image_name = dataset_vsp['image']

    dataset.label = np.array(dataset_vsp['att'])
    assert dataset.label.shape == (len(dataset_vsp['image']), 56)
    dataset.attr_name = dataset_vsp['att_name']

    if reorder:
        dataset.label = dataset.label[:, np.array(group_order)]
        dataset.attr_name = [dataset.attr_name[i] for i in group_order]

    dataset.partition = EasyDict()
    dataset.partition.train = np.array(partition_vsp['train'][0])  # np.array(range(80000))
    dataset.partition.val = np.array(partition_vsp['val'][0])  # np.array(range(80000, 90000))
    dataset.partition.test = np.array(partition_vsp['test'][0])  #np.arange(90000, 100000)  # np.array(range(90000, 100000))
    dataset.partition.trainval = np.array(partition_vsp['trainval'][0]) # np.array(range(90000))

    dataset.weight_train = np.mean(dataset.label[dataset.partition.train], axis=0).astype(np.float32)
    dataset.weight_trainval = np.mean(dataset.label[dataset.partition.trainval], axis=0).astype(np.float32)

    with open(os.path.join(save_dir, 'dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    datapath = '/home/austin/home/dataset/pa100k/data/'
    part_pkl_path = '../partition_pa100_ccf100_56_0427.pkl'
    data_pkl_path = '../dataset_pa100_ccf100_56_0427.pkl'
    partition = pickle.load(open(part_pkl_path, 'rb'))
    dataset = pickle.load(open(data_pkl_path, 'rb'))
    save_dir = './data/VSP/'
    os.makedirs(save_dir, exist_ok= True)
    reoder = False
    generate_data_description(save_dir, reorder=True)
