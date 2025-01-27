from tqdm import tqdm
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from . import attack_utils
from myutils.helpers import test
#import wandb
from datasets import get_dataset
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
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, _ in data_loader:
            # Move inputs to the appropriate device
            inputs = inputs.to(args.device)
            
            # Get predictions from both models
            outputs_S = model_S(inputs)
            outputs_T = model_T(inputs)
            
            # Convert outputs to class predictions
            _, preds_S = torch.max(outputs_S, 1)
            _, preds_T = torch.max(outputs_T, 1)
            
            # Update counts
            total_samples += inputs.size(0)
            consistent_samples += (preds_S == preds_T).sum().item()
            
            # Clear CUDA cache
            torch.cuda.empty_cache()

    # Calculate and return the consistency rate
    consistency_rate = consistent_samples / total_samples
    return consistency_rate

def knockoff(args, T, S, test_loader, tar_acc):
    T.eval()
    S.train()

    sur_data_loader, _ = get_dataset(args.dataset_sur, batch_size = args.batch_size) #_: test dataloader

    if args.opt == 'sgd':
        optS = optim.SGD(S.parameters(), lr=args.lr_clone, momentum=0.9, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, args.epochs)
    else:
        optS = optim.Adam(S.parameters(), lr=args.lr_clone, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, args.epochs)

    results = {'epochs': [], 'accuracy': [], 'accuracy_x': []}
    print('== Constructing Surrogate Dataset ==')
    sur_ds = []
    #重新给训练数据的输出赋值。
    for data, _ in tqdm(sur_data_loader, ncols=128, leave=True):
        data = data.to(args.device)
        Tout = T(data)
        Tout = F.softmax(Tout, dim=1)
        batch = [(a, b) for a, b in zip(data.cpu().detach().numpy(), Tout.cpu().detach().numpy())]
        sur_ds += batch #5w
        #print(len(sur_ds))
    #print(len(sur_ds))
    sur_dataset_loader = torch.utils.data.DataLoader(sur_ds, batch_size=args.batch_size, num_workers=4, shuffle=True)
    #数据sur_ds保存在sur_dataset_loader这个变量名中，每次加载batch_size个。

    print('\n== Training Clone Model ==')

    best_loss = 0
    no_increase_count = 0  # 记录损失增加的次数
    for epoch in range(1, args.epochs+1):
        S.train()
        train_loss, train_acc = attack_utils.train_soft_epoch(S, args.device, sur_dataset_loader, optS)
        test_loss, test_acc = test(S, args.device, test_loader)
        tar_acc_fraction = test_acc/tar_acc
        Fidelity=calculate_consistency_rate(S,T,test_loader)
        print('Epoch: {} Loss: {:.4f} Train Acc: {:.4f}% Test Acc: {:.4f} ({:.4f}x)%\n Fidelity: {:.4f}'.format(epoch, train_loss, train_acc, test_acc, tar_acc_fraction,Fidelity))
        #wandb.log({'Train Acc': train_acc, 'Test Acc': test_acc, "Train Loss": train_loss})
        # 检查当前损失是否高于最佳损失
        if Fidelity < best_loss:
            no_increase_count += 1  # 增加损失增加的计数
        else:
            best_loss = Fidelity
            no_increase_count = 0  # 重置计数器

        print('No Improvement Count:', no_increase_count)

        # 如果连续10次迭代损失都增加，则终止训练
        # if no_increase_count >= 200:
        #     print("Training stopped due to increased loss in 10 consecutive epochs.")
        #     break
        # if schS:
        #     schS.step()
        results['epochs'].append(epoch)
        results['accuracy'].append(test_acc)
        results['accuracy_x'].append(tar_acc_fraction)

    # Fidelity = calculate_consistency_rate(S,T,test_loader)
    # print('Fidelity is:', Fidelity)
    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.model_tgt)
    df = pd.DataFrame(data=results)
    savedir_csv = savedir + 'csv/'
    if not os.path.exists(savedir_csv):
        os.makedirs(savedir_csv)
    df.to_csv(savedir_csv + '/knockoffnets.csv')
    return
