import os
import pickle

import numpy as np
import torch.utils.data as data
from PIL import Image
import random
from tools.function import get_pkl_rootpath
import torchvision.transforms as T


class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2', 'VSP'], \
            f'dataset name {args.dataset} is not exist'

        data_path = get_pkl_rootpath(args.dataset)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name
        attr_label = dataset_info.label

        assert split in dataset_info.partition.keys(), f'split {split} is not exist'
        self.train_or_test = split
        self.dataset = args.dataset
        self.transform = transform
        self.target_transform = target_transform

        self.root_path = dataset_info.root

        self.attr_id = dataset_info.attr_name
        self.attr_num = len(self.attr_id)

        self.img_idx = dataset_info.partition[split]

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]  # default partition 0
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]
        self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label, imgidx = self.img_id[index], self.label[index], self.img_idx[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            if self.train_or_test == 'trainval':
                chioce_list_size1 = []
                chioce_list_size0 = []
                for i in range(8):
                    chioce_list_size1.append(img.size[1] - int(img.size[1] / (10 + i)))
                    chioce_list_size0.append(img.size[0] - int(img.size[0] / (10 + i)))
                transfor_crop = T.Compose([
                    T.RandomApply(
                        [T.RandomCrop((random.choice(chioce_list_size1), random.choice(chioce_list_size0)))],
                        p=0.5),
                ])
                img = transfor_crop(img)

                #待完成 随机crop
                img = self.transform(img)
            else:
                img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_transform = T.Compose([
    #     T.Resize((height, width)),
    #     T.RandomApply([T.ColorJitter(brightness=0.4,
    #                                                    contrast=0.4,
    #                                                    saturation=0.4,
    #                                                    hue=0.2)], p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((height, width)),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     normalize,
    # ])

    # T.RandomApply([T.ColorJitter(brightness=0.4,
    #                              contrast=0.4,
    #                              saturation=0.4,
    #                              hue=0.2)], p=0.5),
    train_transform = T.Compose([
        T.Resize((height, width)),

        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform
