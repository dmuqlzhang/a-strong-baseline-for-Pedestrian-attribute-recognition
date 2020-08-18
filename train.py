import os
import pprint
from collections import OrderedDict, defaultdict
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.AttrDataset import AttrDataset, get_transform
from loss.CE_loss import CEL_Sigmoid
from loss.MCC_loss import multilabel_categorical_crossentropy
from models.base_block import FeatClassifier, BaseClassifier, BaseClassifier_osnet
from models.resnet import resnet50
from models.osnet import osnet_ain_x1_0
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
from test import test_alm
import time
set_seed(605)


def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    log_name = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
    stdout_file = os.path.join(log_dir, f'stdout_{log_name}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = AttrDataset(args=args, split=args.train_split, transform=train_tsfm)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_set = AttrDataset(args=args, split=args.valid_split, transform=valid_tsfm)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = labels.mean(0)

    # backbone = resnet50()
    # classifier = BaseClassifier(nattr=train_set.attr_num)
    backbone = osnet_ain_x1_0(num_classes=56, pretrained=True, loss='softmax')
    classifier = BaseClassifier_osnet(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    #改
    criterion = CEL_Sigmoid(sample_weight)
    # criterion = multilabel_categorical_crossentropy

    param_groups = [{'params': model.module.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.module.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
    if args.training:#训练，每隔1个epoch，验证集评估一次
        best_metric, epoch = trainer(epoch=args.train_epoch,
                                     model=model,
                                     train_loader=train_loader,
                                     valid_loader=valid_loader,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     lr_scheduler=lr_scheduler,
                                     path=save_model_path,
                                     dataset = train_set)

        print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')
    else:#仅评估，不训练
        model_path = args.best_model_path
        saved_state_dict = torch.load(model_path)['state_dicts']
        saved_state_dict = {k.replace('module.', ''): v for k, v in saved_state_dict.items()}
        backbone = osnet_ain_x1_0(num_classes=valid_set.attr_num, pretrained=True, loss='softmax')
        print('make model for only test')
        classifier = BaseClassifier_osnet(nattr=valid_set.attr_num)
        test_model = FeatClassifier(backbone, classifier)
        print("loading model")
        test_model.load_state_dict(saved_state_dict)
        test_model.cuda()
        test_alm(valid_loader, test_model, attr_num=valid_set.attr_num, description=valid_set.attr_id, set='test', threshold= 0.5)



def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path, dataset):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for i in range(epoch):
        #####
        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )
        ##eval in train set
        # test_alm(train_loader, model, attr_num=dataset.attr_num, description=dataset.attr_id, set = 'train')
        # eval in test set
        test_alm(valid_loader, model, attr_num=dataset.attr_num, description=dataset.attr_id,set='test', threshold= 0.5)
        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        lr_scheduler.step(metrics=valid_loss, epoch=i)

        train_result = get_pedestrian_metrics(train_gt, train_probs, threshold=0.5)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs, threshold=0.5)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    args.training = True #False仅评估， True训练期间评估
    main(args)

    # os.path.abspath()


