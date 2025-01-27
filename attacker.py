import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
#import wandb
import random
import numpy as np
import warnings
import torch.nn as nn
warnings.filterwarnings('ignore')
seed = 42
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy("file_system")

torch.cuda.empty_cache()
from myutils.simutils.timer import timer
from myutils.config import parser
from models import get_model
from datasets import get_dataset
from myutils.helpers import test
from attacks import (
    knockoff,
    noise,
    jbda,
    maze,
    knockoff_adaptive,
    PGD,
    #ActiveThief
)

args = parser.parse_args()
#wandb.init(project=args.wandb_project)
run_name = "{}_{}".format(args.dataset, args.attack)
if args.attack == "maze":
    if args.alpha_gan > 0:
        run_name = "{}_{}".format(args.dataset, "pdmaze")
    budget_M = args.budget / 1e6

    if args.white_box:
        grad_est = "wb"
    else:
        grad_est = "nd{}".format(args.ndirs)

    if args.iter_exp > 0:
        run_name += "_{:.2f}M_{}".format(budget_M, grad_est)
    else:
        run_name += "_{:.2f}M_{}_noexp".format(budget_M, grad_est)

#wandb.run.name = run_name
#wandb.run.save()

# Select hardware

if args.device == "gpu":
    import torch.backends.cudnn as cudnn

    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = "cuda"
    gpus = [0, 1, 2, 3]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    #device_ids = [0, 1, 2]
else:
    args.device = "cpu"


def attack():
    savedir = "{}/{}/{}/".format(args.logdir, args.dataset, args.model_tgt)
    #savedir = '{}/{}/{}/True/'.format(args.logdir, args.dataset, args.model_tgt) #adv_training

    train_loader, test_loader = get_dataset(args.dataset, args.batch_size)

    T = get_model(args.model_tgt, args.dataset)  # Target (Teacher)
    S = get_model(args.model_clone, args.dataset)  # Clone  (Student)

    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     # 如果有多个GPU，使用DataParallel来包装模型
    #     S = nn.DataParallel(S.cuda(), device_ids=gpus, output_device=gpus[0])
    
    S = S.to(args.device)

    savepathT = savedir + "T.pt"
    T.load_state_dict(torch.load(savepathT))

    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs!")
    #     # 如果有多个GPU，使用DataParallel来包装模型
    #     T = nn.DataParallel(T.cuda(), device_ids=gpus)

    T = T.to(args.device)
    _, tar_acc = test(T, args.device, test_loader)
    print("* Loaded Target Model *")
    print("Target Accuracy: {:.2f}\n".format(tar_acc))

    if args.attack == "noise":
        noise(args, T, S, test_loader, tar_acc)

    elif args.attack == "knockoff":
        knockoff(args, T, S, test_loader, tar_acc)
    elif args.attack == "knockoff_adaptive":
        knockoff_adaptive(args, T, S, test_loader, tar_acc)
    elif args.attack == "jbda":
        jbda(args, T, S, train_loader, test_loader, tar_acc)
    elif args.attack == "pgd":
        PGD(args, T, S, train_loader, test_loader, tar_acc)
    elif args.attack == "maze":
        maze(args, T, S, train_loader, test_loader, tar_acc)
    # elif args.attack == "AvtiveThief":
    #     ActiveThief(args, T, S, train_loader, test_loader, tar_acc)
    else:
        sys.exit("Unknown Attack {}".format(args.attack))

    savedir_clone = savedir + "clone/"
    if not os.path.exists(savedir_clone):
        os.makedirs(savedir_clone)

    torch.save(S.state_dict(), savedir_clone + "{}.pt".format(args.attack))
    print("* Saved Sur model * ")


def main():
    pid = os.getpid()
    print("pid: {}".format(pid))
    timer(attack)
    exit(0)


if __name__ == "__main__":
    main()
