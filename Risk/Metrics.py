import sys
import os
import time
dir=os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(dir)
sys.path.append(dir)
# my_utils_dir = os.path.join(dir, "myutils")
# print(my_utils_dir)
# sys.path.append(my_utils_dir)
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from myutils.config import parser
from models import get_model
from datasets import get_dataset
from myutils.helpers import test, train_epoch
from torch.autograd import Variable
import os
from numpy.linalg import inv
from numpy.linalg import pinv
import torch.quantization
import torch.nn as nn

os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed = 42  # 或者任何你喜欢的数字
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
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
    label_batch = 50 #尽可能大才能保持稳定
    X_sub_len = X_sub.shape[0]
    num_splits = 1 + int(X_sub_len / label_batch)
    splits = np.array_split(X_sub, num_splits, axis=0)
    #i=0
    for x_sub in splits:
        score_batch = blackbox(to_var(torch.from_numpy(x_sub)))
        #print(score_batch)
        score_batch = F.softmax(score_batch, dim=1)
        score_batch = score_batch.data.cpu().numpy()
        #print(score_batch)
        scores.append(score_batch)
        del score_batch
        torch.cuda.empty_cache()
        #i=i+1
        #print(i)
    scores = np.concatenate(scores)
    print('done labeling')

    y_sub = scores
    return y_sub

def get_outputs(X_sub, blackbox):
    blackbox.eval()
    scores = []
    label_batch = 32
    X_sub_len = X_sub.shape[0]
    num_splits = 1 + int(X_sub_len / label_batch)
    splits = np.array_split(X_sub, num_splits, axis=0)

    for x_sub in splits:
        score_batch = blackbox(to_var(torch.from_numpy(x_sub)))
        #score_batch = F.softmax(score_batch, dim=1)
        score_batch = score_batch.data.cpu().numpy()
        scores.append(score_batch)
    scores = np.concatenate(scores)
    print('done labeling')

    y_sub = scores
    return y_sub

def Metrics_AfterTrainingNTK_SAVE_FISRT(X_train, T ,savedir):
    # 计算训练后的NTK
    # load Initialized model
    T_original = get_model(args.model_tgt, args.dataset)
    T_original = T_original.to(args.device)
    T_original.eval()
    y_pre=get_labels(X_train,T_original)
    print('y_pre.shape: ', y_pre.shape)

    N, K = y_pre.shape
    print('Runing samples:',N)

    parameters_count = 0  # 初始化符合条件的参数计数器

    # 首先计算模型中所有需要梯度的参数的绝对值
    all_params = torch.cat([param.view(-1) for param in T.parameters() if param.requires_grad])

    # 对这些绝对值进行排序
    sorted_params = torch.sort(torch.abs(all_params))[0]

    # 计算保留四分之一参数所需的阈值
    # 注意这里使用了负索引，因为我们要取最大的四分之一
    threshold = sorted_params[int(len(sorted_params) * 3 / 4)]

    for param in T.parameters():
        if param.requires_grad:
            # 计算每个参数中绝对值大于0.1的元素个数
            parameters_count += torch.sum(torch.abs(param) > threshold).item()

    parameters=sum(p.numel() for p in T.parameters() if p.requires_grad)
    print('paremeters_count', parameters_count,parameters)
    grads_np = np.zeros((N, K, parameters), dtype=np.float32) #float32 -->float 16 for CIFAR-100, levit_384, 千万不能这么做，速度会降低几十倍~
    #print(T)
    T.eval()
    #T.train()
    batch_size=16
    # 预先计算所有样本的梯度, 然后再计算NTK
    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        X_train_batch = X_train[batch_start:batch_end]
        X_train_tensor = torch.from_numpy(X_train_batch).to(args.device)
        outputs = T(X_train_tensor)
        # X_train_np=X_train_tensor.cpu().numpy()
        # outputs = get_outputs(X_train_np,T)
        # outputs=torch.tensor(outputs,requires_grad=True)
        # print("1",outputs.shape)
        #T.train()
        for i in range(batch_start, batch_end):
            #print(i)
            batch_i = i - batch_start
            for k in range(K):
                #if (i, k) not in grads_dict:
                    output_grad = torch.zeros_like(outputs)
                    output_grad[batch_i, k] = 1

                    T.zero_grad()
                    outputs.backward(gradient=output_grad, retain_graph=True)
                    grads = []
                    for name, param in T.named_parameters():
                        if param.grad is not None:
                            # mask = torch.abs(param) > threshold
                            # important_grads = torch.masked_select(param.grad, mask)
                            # grads.append(important_grads)
                            grads.append(param.grad.view(-1))
                            #print(grads)
                        else:
                            # 如果梯度不存在，则添加一个全零的张量，其形状与参数相同
                            grads.append(torch.zeros_like(param).view(-1))

                    if grads:
                        #grad_np = torch.cat([g.view(-1) for g in grads]).cpu().numpy()

                        grad_np = torch.cat(grads).cpu().numpy()
                        #print(grad_np.shape)
                        grads_np[i, k] = grad_np
        del outputs
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    print(1)
    # grads_mean = np.mean(np.abs(grads_np), axis=(0, 1))
    # print(2)
    # 对神经元按梯度绝对值平均进行排序，并获取前20%的神经元索引
    # top_percent_indices = np.argsort(-grads_mean)[:int(0.5 * parameters)]
    # print(3)
    # grads_np = grads_np[:, :, top_percent_indices]
    
    print('grads_np.shape:',grads_np.shape)
    print('grads_np.type:',grads_np.dtype)

    # # 初始化NTK矩阵(在CPU上)
    grads_matrix = grads_np.reshape(N * K, -1)
    ntk_matrix = np.dot(grads_matrix, grads_matrix.T)
    print(ntk_matrix.shape)
    ntk=ntk_matrix

    # ntk_matrix = np.zeros((N * K, N * K), dtype=np.float32)
    # print('CPU')
    # for i in range(N * K):
    #     print(i)
    #     for j in range(i, N * K):
    #         # 提取对应的梯度
    #         grads1 = grads_np[i // K, i % K]
    #         grads2 = grads_np[j // K, j % K]
    #         # 计算NTK矩阵的元素
    #         ntk_value = np.dot(grads1, grads2)
    #         ntk_matrix[i, j] = ntk_value
    #         ntk_matrix[j, i] = ntk_value
    # ntk=ntk_matrix
    # print(ntk_matrix[1,1])

    print(2)

    #Computer change of y
    T.eval()
    y_pre_train = get_labels(X_train, T)
    #print(y_pre_train)
    y_change = y_pre_train - y_pre
    y_change = np.reshape(y_change, (-1, 1))
    print('y_change.shape', y_change.shape)

    #Computer metric_recovery complexity
    #ntk=ntk_matrix.cpu().numpy()
    torch.cuda.empty_cache()

    # savedirT=savedir + "NTK.npy"
    # np.save(savedirT, ntk)

    eigenvalues, eigenvectors = np.linalg.eig(ntk)
    Eigenvalue_min = np.min(eigenvalues)
    print('Eigenvalue_min:',Eigenvalue_min)
    if np.all(eigenvalues > 0):
        print("Matrix is positive definite")
        # eigenvalues=np.maximum(eigenvalues, 30)
        # ntk= eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        ntk_inv=pinv(ntk)
    else:
        print("Matrix is not positive definite")
        # positive_eigenvalues = np.maximum(eigenvalues, 0.1)
        # ntk_inv = eigenvectors @ np.diag(1.0 / positive_eigenvalues) @ eigenvectors.T
        eigenvalues=np.maximum(eigenvalues, 0.5)
        #eigenvalues = np.where(eigenvalues < 0, 0.5, eigenvalues)
        #eigenvalues=np.maximum(eigenvalues, np.abs(Eigenvalue_min))
        #eigenvalues = np.real(eigenvalues)
        ntk= eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        ntk_inv=pinv(ntk)
        eigenvalues1 = np.linalg.eigvals(ntk)
        if np.all(eigenvalues1 > 0):
            print("after Matrix is positive definite")
        else:
            print("after Matrix is not positive definite")
    
    #ntk_inv = pinv(ntk)
    Metrics_RC = np.dot(np.dot(y_change.T, ntk_inv), y_change)
    print('Metrics_RC:',Metrics_RC)
    print('Metrics_RC_abs:',np.abs(Metrics_RC))

    # #Computer metric_conditional number
    # Metrics_cn = np.linalg.cond(ntk)
    # print('Metrics_cn',Metrics_cn)

    # #Computer eignevalues
    # eigenvalues = np.linalg.eigvals(ntk)
    # Eigenvalue_max = np.max(eigenvalues)
    # Eigenvalue_min = np.min(eigenvalues)
    # print('Max and Min of eigenvalues:', Eigenvalue_max, Eigenvalue_min)
    
    # # 使用奇异值分解获取奇异值
    # u, s, vh = np.linalg.svd(ntk)

    # # 计算核心范数（nuclear norm）
    # nuclear_norm = np.sum(s)

    # print('nuclear_norm:', nuclear_norm)
    Metrics_cn= Eigenvalue_max= Eigenvalue_min= nuclear_norm=0
    return [Metrics_RC, Metrics_cn, Eigenvalue_max, Eigenvalue_min,nuclear_norm]

def Select_Samples(X_train,y_predict,y_train,large,small):
    # # # 计算 margins
    # sorted_predictions = np.sort(y_predict, axis=1)
    # margins = sorted_predictions[:, -1] - sorted_predictions[:, -2]

    # # 计算每个样本的预测类别
    # predicted_classes = np.argmax(y_predict, axis=1)

    # # 为每个类别选择10个具有最小 margin 的样本
    # selected_indices = []

    # num_classes = y_predict.shape[1]  # 总类别数
    # num_samples_to_select_max = int(large/num_classes)   # 每个类别要选择的样本数
    # num_samples_to_select_min = int(small/num_classes)   # 每个类别要选择的样本数
    # print(num_samples_to_select_max,num_samples_to_select_min)

    # for class_idx in range(num_classes):
    #     # 获取当前类别的样本索引
    #     indices_of_class = np.where(predicted_classes == class_idx)[0]
        
    #     # 计算这些样本的 margins
    #     margins_of_class = margins[indices_of_class]
        
    #     # 为当前类别选择n个具有最小 margin 的样本
    #     smallest_margins_indices_of_class = indices_of_class[np.argsort(margins_of_class)[:num_samples_to_select_min]]
    #     # 为当前类别选择n个具有最大 margin 的样本
    #     largest_margins_indices_of_class = indices_of_class[np.argsort(-margins_of_class)[:num_samples_to_select_max]]

    #     selected_indices.extend(smallest_margins_indices_of_class)
    #     selected_indices.extend(largest_margins_indices_of_class)

    
    # # 使用选定的索引从 X_train 中提取样本
    # X_train= X_train[selected_indices]
    # y_train = y_train[selected_indices]


    # Compute margin
    sorted_predictions = np.sort(y_predict, axis=1)
    margins = sorted_predictions[:, -1] - sorted_predictions[:, -2]
    # Get indices of the x samples with the smallest margins
    smallest_margin_indices = np.argsort(margins)[:small]
    
    # X_train = X_train[smallest_margin_indices]
    # y_train = y_train[smallest_margin_indices]

    largest_margin_indices = np.argsort(-margins)[:large] #最大的
    combined_indices = np.union1d(smallest_margin_indices, largest_margin_indices)
    X_train = X_train[combined_indices]
    y_train = y_train[combined_indices]

    
    # print('Select X_train.shape: ',X_train.shape)

    # Dataset Distillation
    # datadir = "./{}/{}/".format(args.logdir, args.dataset)
    # xdir=datadir+'CIFAR10_10/images_best.pt'
    # #xdir=datadir+'images_best.pt'
    # images_tensor = torch.load(xdir)
    # #labels_tensor = torch.load("/mnt/data/labels_best.pt")
    # # Convert tensors to numpy arrays
    # X_train = images_tensor.numpy()
    print('X_train.shape: ',X_train.shape)
    #labels_np = labels_tensor.numpy()
    return X_train,y_train

def main():
    # Model+Dataset
    #Metrics_RC()
    start_time = time.time()
    T = get_model(args.model_tgt, args.dataset)
    #savedir = "./{}/{}/{}/True/".format(args.logdir, args.dataset, args.model_tgt)
    savedir = "./{}/{}/{}/".format(args.logdir, args.dataset, args.model_tgt)
    print(savedir)
    savepathT = savedir + "T.pt"
    T.load_state_dict(torch.load(savepathT))
    T = T.to(args.device)
    total_params = sum(p.numel() for p in T.parameters() if p.requires_grad)
    print('#Params:',total_params)
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
        data_list.append(data.cpu().numpy())
        labels_list.append(labels.cpu().numpy())
    X_train = np.concatenate(data_list, axis=0)
    print('X_train.shape: ',X_train.shape)
    y_train = np.concatenate(labels_list, axis=0)
    y_predict = get_labels(X_train,T)

    # # #random select
    # random_indices = np.random.choice(len(X_train), 400, replace=False)
    # #print(random_indices)
    # X_train1 = X_train[random_indices]
    # y_train1 = y_train[random_indices]
    # print("--------Random--------------------------")
    # [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_AfterTrainingNTK_SAVE_FISRT(X_train1,T,savedir)

    # [X_train2,y_train2]=Select_Samples(X_train,y_predict,y_train,large=100,small=0)
    # print('X_.shape: ',X_train2.shape)
    # print("--------Max 100--------------------------")
    # [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_AfterTrainingNTK_SAVE_FISRT(X_train2,T,savedir)

    # [X_train3,y_train3]=Select_Samples(X_train,y_predict,y_train,large=0,small=100)
    # print('X_.shape: ',X_train3.shape)
    # print("--------Small 100--------------------------")
    # [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_AfterTrainingNTK_SAVE_FISRT(X_train3,T,savedir)

    [X_train4,y_train4]=Select_Samples(X_train,y_predict,y_train,large=50,small=50)
    print('X_.shape: ',X_train4.shape)
    print("--------Max 50 + small 50--------------------------")
    [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_AfterTrainingNTK_SAVE_FISRT(X_train4,T,savedir)

    # [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_AfterTrainingNTK_SAVE_And_Compter(X_train,T,savedir)
    # [Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_AfterTrainingNTK_SAVE_FISRT(X_train,T,savedir)


    #[Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_Reduce_Gradient_CPU(X_train,T,savedir)
    #[Metrics_RC, Metrics_cn, Eigenvalue_max, Eignevalue_min, nuclear_norm]=Metrics_approximate(X_train,y_train,T)
    #NTK=NTK.cpu().numpy()
    #print('size of NTK:', NTK.shape)

    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    print(f"Time: {execution_time} s")
    print(f"Time: {hours} h {minutes} mins {seconds} s")

if __name__ == "__main__":
    main()