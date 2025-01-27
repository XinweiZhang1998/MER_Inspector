from tqdm import tqdm
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from myutils import test
import pandas as pd
from .attack_utils import (
    kl_div_logits,
    generate_images,
    sur_stats,
    zoge_backward,
    gradient_penalty,
)
import openpyxl
#import wandb
from models import get_model
import pandas as pd
from myutils.simutils import logs
import itertools
from myutils.config import parser
args = parser.parse_args()

if args.device == 'gpu':
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = 'cuda'
else:
    args.device = 'cpu'

def calculate_consistency_rate(model_S, model_T, data_loader):
    """
    Calculate the consistency rate between two models (S and T) using a data loader.
    
    Parameters:
        model_S, model_T (torch.nn.Module): The two models to compare.
        data_loader (torch.utils.data.DataLoader): The data loader providing input samples.
        
    Returns:
        float: The consistency rate between the two models.
    """
    model_S.eval()  # Set the model to evaluation mode
    model_T.eval()  # Set the model to evaluation mode
    
    total_samples = 0
    consistent_samples = 0
    
    for inputs, _ in data_loader:
        # Move inputs to the appropriate device
        inputs = inputs.to(args.device)  # Assume `device` is defined elsewhere
        
        # Get predictions from both models
        outputs_S = model_S(inputs)
        outputs_T = model_T(inputs)
        
        # Convert outputs to class predictions
        _, preds_S = torch.max(outputs_S, 1)
        _, preds_T = torch.max(outputs_T, 1)
        
        # Update counts
        total_samples += inputs.size(0)
        consistent_samples += (preds_S == preds_T).sum().item()
    
    # Calculate and return the consistency rate
    consistency_rate = consistent_samples / total_samples
    return consistency_rate

tanh = nn.Tanh()
import matplotlib

matplotlib.use("Agg")

def margin(X_sub1,y_sub1):
    # 返回平均margin值，将logits改写为probaility
    X_sub = X_sub1.cpu().numpy()
    #y_sub = y_sub1.cpu().numpy()
    #print('y_sub.shape_before:',y_sub.shape)
    margin_values = []
    for i in range(X_sub.shape[0]):
        # 找到样本所在的类别
        #print(y_sub[i])
        score_batch = F.softmax(y_sub1, dim=1)
        y_sub = score_batch.data.cpu().numpy()
        #print(y_sub.shape)
        y_pred = np.argmax(y_sub[i])
        #print(y_pred)
        # 计算margin值
        margin_i = y_sub[i][y_pred] - np.max(np.delete(y_sub[i], y_pred))
        margin_values.append(margin_i)
    return np.mean(margin_values)

def maze(args, T, S, train_loader, test_loader, tar_acc):

    G = get_model(args.model_gen, args.dataset, latent_dim=args.latent_dim)
    G.to(args.device)
    D = get_model(args.model_dis, args.dataset)
    D.to(args.device)
    T.eval(), S.train(), G.train(), D.train()
    schD = schS = schG = None
    df1 = pd.DataFrame()
    budget_per_iter = args.batch_size * (
        (args.iter_clone - 1) + (1 + args.ndirs) * args.iter_gen
    )
    iter = int(args.budget / budget_per_iter)

    if args.opt == "sgd":
        optS = optim.SGD(
            S.parameters(), lr=args.lr_clone, momentum=0.9, weight_decay=5e-4
        )
        optG = optim.SGD(
            G.parameters(), lr=args.lr_gen, momentum=0.9, weight_decay=5e-4
        )
        optD = optim.SGD(
            D.parameters(), lr=args.lr_dis, momentum=0.9, weight_decay=5e-4
        )

        schS = optim.lr_scheduler.CosineAnnealingLR(optS, iter, last_epoch=-1)
        schG = optim.lr_scheduler.CosineAnnealingLR(optG, iter, last_epoch=-1)
        schD = optim.lr_scheduler.CosineAnnealingLR(optD, iter, last_epoch=-1)

    else:
        optS = optim.Adam(S.parameters(), lr=args.lr_clone)
        optG = optim.Adam(G.parameters(), lr=args.lr_gen)
        optD = optim.Adam(D.parameters(), lr=args.lr_dis)

    print("\n== Training Clone Model ==")

    lossG = lossG_gan = lossG_dis = lossD = cs = mag_ratio = torch.tensor(0.0)
    query_count = 0
    log = logs.BatchLogs()
    start = time.time()
    test_noise = torch.randn((5 * 5, args.latent_dim), device=args.device)
    results = {"queries": [], "accuracy": [], "accuracy_x": []}
    ds = []  # dataset for experience replay

    #当有初始样本时；
    if args.alpha_gan > 0:
        assert args.num_seed > 0  # We need to have seed examples to train gan  和JBDA一样
        print("\nQuerying the target model with initial Dataset")
        data_loader_real = torch.utils.data.DataLoader(
            train_loader.dataset, batch_size=args.num_seed, shuffle=True
        )
        data_loader_real = itertools.cycle(data_loader_real)
        x_seed, y_seed = next(data_loader_real)
        x_seed = x_seed.to(args.device)
        Tout = T(x_seed)
        batch = [
            (a, b)
            for a, b in zip(x_seed.cpu().detach().numpy(), Tout.cpu().detach().numpy())
        ]
        ds += batch
        print("Done!")

        # Build data loader with seed examples
        seed_ds_batch = [
            (a, b)
            for a, b in zip(
                x_seed.cpu().detach().numpy(), y_seed.cpu().detach().numpy()
            )
        ]
        seed_ds = []
        for _ in range(10):
            seed_ds += seed_ds_batch
        data_loader_real = torch.utils.data.DataLoader(
            seed_ds, batch_size=args.batch_size, num_workers=4, shuffle=True
        )
        # train_loader_seed = create_seed_loader(args, x_seed.cpu().numpy(), y_seed.cpu().numpy())
        data_loader_real = itertools.cycle(data_loader_real)

    for p in T.parameters():
        p.requires_grad = False

    pbar = tqdm(range(1, iter + 1), ncols=80, disable=args.disable_pbar, leave=False)
    for i in pbar:

        ###########################
        # (1) Update Generator
        ###########################
        # margin_all1=[]
        for g in range(args.iter_gen):
            z = torch.randn((args.batch_size, args.latent_dim), device=args.device)
            x, x_pre = G(z)
            optG.zero_grad()

            if args.white_box:
                Tout = T(x)
                Sout = S(x)
                lossG_dis = -kl_div_logits(args, Tout, Sout)
                (lossG_dis).backward(retain_graph=True)
                # x1=x.cpu().numpy()
                # Tout1=Tout.cpu().numpy()
                # margin_all1.append(margin(x1, Tout1))
            else:
                lossG_dis, cs, mag_ratio = zoge_backward(args, x_pre, x, S, T)

            if args.alpha_gan > 0:
                lossG_gan = D(x)
                lossG_gan = -lossG_gan.mean()
                (args.alpha_gan * lossG_gan).backward(retain_graph=True)

            lossG = lossG_dis + (args.alpha_gan * lossG_gan)
            optG.step()
        # avg_margin= np.mean(margin_all1)
        # print("marghin",avg_margin)

        log.append_tensor(
            ["Gen_loss", "Gen_loss_dis", "Gen_loss_gan", "cs", "mag_ratio"],
            [lossG, lossG_dis, lossG_gan, cs, mag_ratio],
        )

        ############################
        # (2) Update Clone network
        ###########################

        for c in range(args.iter_clone):
            with torch.no_grad():
                if c != 0:  # reuse x from generator update for c == 0
                    z = torch.randn(
                        (args.batch_size, args.latent_dim), device=args.device
                    )
                    x, _ = G(z)
                x = x.detach()
                Tout = T(x)

            Sout = S(x)
            lossS = kl_div_logits(args, Tout, Sout)
            optS.zero_grad()
            lossS.backward()
            optS.step()

            ############################
            ## (3) Update Critic ##  (ONLY for partial data setting)
            ############################
            # We assume iter_clone == iter_critic and share the training loop of the Clone for Critic update
            # This saves an extra evaluation of the generator
            if args.alpha_gan > 0:
                x_real = next(data_loader_real)[0]
                if x_real.size(0) < args.batch_size:
                    x_real = next(data_loader_real)[0]
                x_real = x_real.to(args.device)

                lossD_real = D(x_real)
                lossD_real = -lossD_real.mean()
                lossD_fake = D(x)
                lossD_fake = lossD_fake.mean()

                # train with gradient penalty
                gp = gradient_penalty(x.data, x_real.data, D)
                lossD = lossD_real + lossD_fake + args.lambda1 * gp
                optD.zero_grad()
                lossD.backward()
                optD.step()

        _, max_diff, max_pred = sur_stats(Sout, Tout)
        log.append_tensor(
            ["Sur_loss", "Dis_loss", "Max_diff", "Max_pred"],
            [lossS, lossD, max_diff, max_pred],
        )

        ############################
        # (4) Experience Replay
        ###########################

        # if args.gen_dataset:
        # Store the last batch for experience replay
        batch = [
            (a, b)
            for a, b in zip(x.cpu().detach().numpy(), Tout.cpu().detach().numpy())
        ]

        ds += batch
        gen_train_loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True
        )
        gen_train_loader_iter = itertools.cycle(gen_train_loader)

        lossS_exp = torch.tensor(0.0, device=args.device)
        margin_all=[]
        for c in range(args.iter_exp): #10次额外的
            x_prev, T_prev = next(gen_train_loader_iter)
            if x_prev.size(0) < args.batch_size:
                break
            x_prev, T_prev = x_prev.to(args.device), T_prev.to(args.device)
            margin_all.append(margin(x_prev, T_prev))
            #print('X_prev size:', x_prev.shape)
            S_prev = S(x_prev)
            lossS = kl_div_logits(args, T_prev, S_prev)
            optS.zero_grad()
            lossS.backward()
            optS.step()
            lossS_exp += lossS
        print('Margin:', np.mean(margin_all))
        avg=np.mean(margin_all)
        df_temp = pd.DataFrame([avg], columns=['Margin'])
        df1 = pd.concat([df1, df_temp])

        if args.iter_exp:
            lossS_exp /= args.iter_exp
        log.append_tensor(["Sur_loss_exp"], [lossS_exp])

        query_count += budget_per_iter

        if (
            query_count % args.log_iter < budget_per_iter
            and query_count > budget_per_iter
        ) or i == iter:
            log.flatten()
            _, log.metric_dict["Sur_acc"] = test(S, args.device, test_loader)
            tar_acc_fraction = log.metric_dict["Sur_acc"] / tar_acc
            log.metric_dict["Sur_acc(x)"] = tar_acc_fraction

            metric_dict = log.metric_dict
            generate_images(args, G, test_noise, "G")
            pbar.clear()
            time_100iter = int(time.time() - start)
            # for param_group in optS.param_groups:
            #    print("learning rate S ", param_group["lr"])

            iter_M = query_count / 1e6
            print(
                "Queries: {:.2f}M Losses: Gen {:.2f} Sur {:.2f} Acc: Sur {:.2f} ({:.2f}x) time: {: d}".format(
                    iter_M,
                    metric_dict["Gen_loss"],
                    metric_dict["Sur_loss"],
                    metric_dict["Sur_acc"],
                    tar_acc_fraction,
                    time_100iter,
                )
            )

            #wandb.log(log.metric_dict)
            results["queries"].append(iter_M)
            results["accuracy"].append(metric_dict["Sur_acc"])
            results["accuracy_x"].append(tar_acc_fraction)

            log = logs.BatchLogs()
            S.train()
            start = time.time()

        loss_test, _ = test(S, args.device, test_loader)

        if schS:
            schS.step()
        if schG:
            schG.step()
        if schD and args.alpha_gan > 0:
            schD.step()

    Fidelity = calculate_consistency_rate(S,T,test_loader)
    print('Fidelity is:', Fidelity)
    df1.to_excel('output.xlsx', index=False)
    savedir = "{}/{}/{}/".format(args.logdir, args.dataset, args.model_tgt)
    savedir_csv = savedir + "csv/"
    df = pd.DataFrame(data=results)
    budget_M = args.budget / 1e6
    if not os.path.exists(savedir_csv):
        os.makedirs(savedir_csv)
    if args.alpha_gan > 0:
        df.to_csv(savedir_csv + "/pdmaze_{:.2f}M.csv".format(budget_M))
    else:
        df.to_csv(savedir_csv + "/maze_{:.2f}M.csv".format(budget_M))
    return

