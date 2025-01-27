import torch
import math
import sys
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
#import wandb
import numpy as np
eps=1e-7
from myutils.simutils import logs

#from simutils import logs
from torch.autograd import Variable
from torch import autograd
import itertools
import torch.optim as optim
import random
#import kornia
import copy
import seaborn as sns
import time
tanh = nn.Tanh()
import pandas as pd
#from cleverhans.torch.attacks import fast_gradient_method, projected_gradient_descent


# def projected_gradient_descent(model, data, eps, step_alpha, num_steps, norm_type):
#     """Perform projected gradient descent on the input data."""
#     # Start with the original data
#     adv_data = data.clone().detach().requires_grad_(True).to(data.device)
#     target = model(data).max(1)[1]  # Assuming the original model prediction is the target

#     for _ in range(num_steps):
#         # Forward pass
#         output = model(adv_data)
#         loss = F.nll_loss(output, target)  # Assuming using negative log likelihood loss
#         # Backward pass
#         loss.backward()
#         # Generate perturbation
#         perturbation = step_alpha * adv_data.grad.data.sign()
#         # Update adversarial data
#         adv_data = adv_data + perturbation
#         adv_data = torch.min(torch.max(adv_data, data - eps), data + eps)
#         adv_data = torch.clamp(adv_data, 0, 1).detach().requires_grad_(True)

#     return adv_data

criterion = nn.CrossEntropyLoss()

def pgd_attack(model, images, labels, eps, alpha, iters):
    # 图片的原始副本
    original_images = images.clone().detach()

    # 对每个像素进行扰动
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        
        model.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        # 有界扰动
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach()

    return images

def train_epoch(model, device, train_loader, opt, args, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable = disable_pbar, leave=False)):
        data = torch.tensor(data).to(device)
        target = torch.tensor(target).to(device)
        #data, target = data.to(device), target.to(device)
        # print("Size of data1:", data.size())
        # print("Size of target1:", target.size())
        opt.zero_grad()
        output = model(data)
        #loss = criterion(output, target.long())

        loss = criterion(output, target) 

        if args.adv_train:
            #niter = 5
            eps = 0.1
            alpha = 0.01
            iters = 3
            adv_data = pgd_attack(model, data, target, eps, alpha, iters)

            #data_adv = projected_gradient_descent(model, data, args.eps_adv, args.eps_adv/niter, niter, np.inf)
            output_adv = model(data) 
            #loss += criterion(output_adv, target)  
            loss_adv = criterion(output_adv, target)
            loss = (loss + loss_adv) / 2

        loss.backward()
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_samples += target.size(0)  # 累加本批次的样本数

    if total_samples > 0:
        train_loss /= total_samples
        train_acc = correct * 100. / total_samples

    # train_loss /= total_samples
    # train_acc = correct * 100. / total_samples

    # train_loss /= len(train_loader)
    # train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            #data, target = next(iter(test_loader))
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += target.size(0)

    if total_samples > 0:
        test_loss /= total_samples
        test_acc = correct * 100. / total_samples
    
    model.train()
    return test_loss, test_acc