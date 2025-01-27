import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from utils.config import parser
from models import get_model
from datasets import get_dataset
from utils.helpers import test, train_epoch
from torch.autograd import Variable
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed = 42  # 或者任何你喜欢的数字
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parser.parse_args()
if args.device == "gpu":
    import torch.backends.cudnn as cudnn

    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = "cuda"
else:
    args.device = "cpu"
def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def get_labels(X_sub, blackbox):
    blackbox.eval()
    scores = []
    label_batch = 64
    X_sub_len = X_sub.shape[0]
    num_splits = 1 + int(X_sub_len / label_batch)
    splits = np.array_split(X_sub, num_splits, axis=0)

    for x_sub in splits:
        score_batch = blackbox(to_var(torch.from_numpy(x_sub)))
        score_batch = F.softmax(score_batch, dim=1)
        score_batch = score_batch.data.cpu().numpy()
        scores.append(score_batch)
    scores = np.concatenate(scores)
    print('done labeling')

    y_sub = scores
    return y_sub

def Metrics(X_train,T):
    # load Initialized model
    T_original = get_model(args.model_tgt, args.dataset)
    T_original = T_original.to(args.device)
    T_original.eval()
    y_pre=get_labels(X_train,T_original)
    # for name, param in T_original.named_parameters():
    #     print(name, param.data)
    print('y_pre.shape: ', y_pre.shape)
    X_train =X_train[:1000]
    y_pre = y_pre[:1000]
    N, K = y_pre.shape
    ntk_matrix = torch.zeros(N * K, N * K)

    # 计算NTK矩阵的一个子集
    for i in range(N):
        x1 = X_train[i:i + 1]
        for j in range(N):
            x2 = X_train[j:j + 1]
            # 前向传播得到输出
            x1 = torch.tensor(x1,requires_grad=True)
            x1 = x1.to(args.device)
            #T_original.eval()
            output1 = T_original(x1)
            x2 = torch.tensor(x2,requires_grad=True)
            x2 = x2.to(args.device)
            output2 = T_original(x2)
            # output1 = get_labels(x1,T_original)
            # output2 = get_labels(x2,T_original)
            #output1 = torch.tensor(output1, requires_grad=True)
            #output2 = torch.tensor(output2, requires_grad=True)
            T_original.eval()
            for k in range(K):  # 对于每个输出类别
                # 手动设置输出的grad属性
                output1_grad = torch.zeros_like(output1)
                output1_grad[0, k] = 1

                # 使用backward计算梯度
                #output1.requires_grad = True
                #print('output1.requires_grad:',output1.requires_grad)

                #T_original.train()
                T_original.zero_grad()
                output1.backward(gradient=output1_grad, retain_graph=True)

                # 获取梯度
                grads1 = []
                for name, param in T_original.named_parameters():
                    #print('param.requires_grad:',param.requires_grad)
                    if param.grad is not None:
                        grads1.append(param.grad.view(-1))
                    else:
                        print(f"No gradient for {name}, skipping!")
                    #grads1.append(param.grad.view(-1))
                grads1 = torch.cat(grads1)

                for l in range(K):
                    # 手动设置输出的grad属性
                    output2_grad = torch.zeros_like(output2)
                    output2_grad[0, l] = 1

                    # 使用backward计算梯度
                    T_original.zero_grad()
                    output2.backward(gradient=output2_grad, retain_graph=True)

                    # 获取梯度
                    grads2 = []
                    for name, param in T_original.named_parameters():
                        grads2.append(param.grad.view(-1))
                    grads2 = torch.cat(grads2)

                    # 更新NTK矩阵
                    ntk_matrix[i * 10 + k, j * 10 + l] = torch.dot(grads1, grads2)
    #Computer change of y
    T.eval()
    y_pre_train = get_labels(X_train, T)
    y_change = y_pre_train - y_pre
    y_change = np.reshape(y_change, (-1, 1))
    print('y_change.shape', y_change.shape)

    #Computer metric_recovery complexity
    ntk=ntk_matrix.cpu().numpy()
    Metrics_RC = np.dot(np.dot(y_change.T, ntk), y_change)
    print('Metrics_RC:',Metrics_RC)

    #Computer metric_conditional number
    Metrics_cn = np.linalg.cond(ntk)
    print('Metrics_cn',Metrics_cn)

    #Computer eignevalues
    eigenvalues = np.linalg.eigvals(ntk)
    Eigenvalue_max = np.max(eigenvalues)
    Eigenvalue_min = np.min(eigenvalues)
    print('Max and Min of eigenvaluses:', Eigenvalue_max, Eigenvalue_min)

    return [Metrics_RC, Metrics_cn, Eigenvalue_max, Eigenvalue_min]

def main():
    # Model+Dataset
    #Metrics_RC()
    T = get_model(args.model_tgt, args.dataset)
    savedir = "../{}/{}/{}/".format(args.logdir, args.dataset, args.model_tgt)
    savepathT = savedir + "T.pt"
    T.load_state_dict(torch.load(savepathT))
    T = T.to(args.device)
    # for name, param in T.named_parameters():
    #     print(name, param.data)
    #Dataset_name = 'cifar10'
    train_loader, test_loader = get_dataset(args.dataset, args.batch_size)
    _, tar_acc = test(T, args.device, test_loader)
    print("* Loaded Target Model *")
    print("Target Accuracy: {:.2f}\n".format(tar_acc))
    data_list = []
    labels_list = []
    for data, labels in train_loader:
        data_list.append(data.numpy())
        labels_list.append(labels.numpy())
    X_train = np.concatenate(data_list, axis=0)
    print('X_train.shape: ',X_train.shape)
    y_train = np.concatenate(labels_list, axis=0)

    [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min]=Metrics(X_train,T)
    #NTK=NTK.cpu().numpy()
    #print('size of NTK:', NTK.shape)

if __name__ == "__main__":
    main()