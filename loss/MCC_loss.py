import torch
import numpy as np
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    tmp1 = y_true.detach().cpu()
    tmp2 = y_pred.detach().cpu()
    y_pred = tmp1
    y_true = tmp2
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    # y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    # y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    y_pred_neg = torch.cat((y_pred_neg, zeros), axis=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), axis=-1)
    # neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    # pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    neg_loss = torch.log(torch.sum(torch.exp(y_pred_neg)))
    pos_loss = torch.log(torch.sum(torch.exp(y_pred_pos)))
    loss = (neg_loss + pos_loss).cuda().requires_grad_()
    return loss