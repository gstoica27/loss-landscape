"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm

from torch.autograd.variable import Variable
from data.shapeworld.utils import *


def cudafy_elems(*args, use_cuda=True):
    if use_cuda:
        return [elem.cuda() for elem in args]
    else:
        return args
    

def eval_loss(net, criterion, dataset, split_name, config, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples

    if use_cuda:
        net.cuda()
    net.eval()

    for idx, batch in tqdm(enumerate(dataset.epoch(
        n=config['batch_size'],
        mode=split_name,
        include_model=config['dataset']['include_model']
    ))):
        data = prepare_data(
            batch,
            config['dataset']['description'],
            id2language=None,
            artificial_and=config.get('artificial_and', False)
        )

        batch_images = torch.from_numpy(
            np.array(data['images'], np.float32)
        )
        batch_labels = np.array(data['labels'], dtype=np.int).reshape(-1)
        queries = data['queries']

        batch, labels = cudafy_elems(batch_images, torch.from_numpy(batch_labels), use_cuda=use_cuda)
        probs = net(queries, batch)
        loss = criterion(probs, labels)
        total += len(batch_images)
        total_loss += loss.item() #* len(batch_images)
        
        predictions = torch.max(probs.data, 1)[0]
        correct += predictions.eq(labels).sum().item()


    # with torch.no_grad():
    #     if isinstance(criterion, nn.CrossEntropyLoss):
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             batch_size = inputs.size(0)
    #             total += batch_size
    #             inputs = Variable(inputs)
    #             targets = Variable(targets)
    #             if use_cuda:
    #                 inputs, targets = inputs.cuda(), targets.cuda()
    #             outputs = net(inputs)
    #             loss = criterion(outputs, targets)
    #             total_loss += loss.item()*batch_size
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += predicted.eq(targets).sum().item()
    #
    #     elif isinstance(criterion, nn.MSELoss):
    #         for batch_idx, (inputs, targets) in enumerate(loader):
    #             batch_size = inputs.size(0)
    #             total += batch_size
    #             inputs = Variable(inputs)
    #
    #             one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
    #             one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
    #             one_hot_targets = one_hot_targets.float()
    #             one_hot_targets = Variable(one_hot_targets)
    #             if use_cuda:
    #                 inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
    #             outputs = F.softmax(net(inputs))
    #             loss = criterion(outputs, one_hot_targets)
    #             total_loss += loss.item()*batch_size
    #             _, predicted = torch.max(outputs.data, 1)
    #             correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total
