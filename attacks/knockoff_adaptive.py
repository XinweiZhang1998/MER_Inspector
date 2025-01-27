import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from myutils.helpers import test
import torch.nn.functional as F
from datasets import get_dataset
from tqdm import tqdm

from attacks import attack_utils

num_classes_dict={
    'fashionmnist':10,
    'cifar10':10,
    'cifar100':100,
    'svhn':10,
    'gtsrb':43
}

def get_labels(X_sub, blackbox):
    scores = []
    label_batch = 64
    X_sub_len = X_sub.shape[0]
    num_splits = 1 + int(X_sub_len / label_batch)
    splits = np.array_split(X_sub, num_splits, axis=0)

    for x_sub in splits:
        score_batch = blackbox(to_var(torch.from_numpy(x_sub)))
        #print('type(score_batch)',score_batch.size())
        score_batch = F.softmax(score_batch, dim=1)
        score_batch = score_batch.data.cpu().numpy()
        scores.append(score_batch)
    scores = np.concatenate(scores)
    #print('done labeling')

    y_sub = scores
    #print(scores)
    return y_sub

def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)



def knockoff_adaptive(args, T, S,  test_loader, tar_acc):
    # 导出surrgate dataset的全部数据以供挑选
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset),
                                              shuffle=False, num_workers=2)
    images, labels = next(iter(trainloader))
    images_np = images.numpy()
    labels_np = labels.numpy()  # TRUE VALUE
    X_sub = images_np
    print('sur_X_sub.shape:', X_sub.shape)
    y_sub = get_labels(X_sub, T)
    num_classes = 10

    # train_loader = torch.myutils.data.DataLoader(train_loader.dataset, batch_size=1000, shuffle=True)
    # num_classes = num_classes_dict[args.dataset]
    # data_iter = iter(train_loader)
    # #X_sub和y_sub是初始数据集
    # X_sub, _ = data_iter.__next__()
    # X_sub = X_sub.numpy()
    # print('X_sub.shape:',X_sub.shape)
    # y_sub = get_labels(X_sub, T)

    X_sub_initial = X_sub[np.random.choice(X_sub.shape[0], 128)]
    y_sub_initial = get_labels(X_sub_initial, T)

    Sur_initial = [(a, b) for a, b in zip(X_sub_initial, y_sub_initial)]
    sur_dataset_loader = torch.utils.data.DataLoader(Sur_initial, batch_size=args.batch_size, num_workers=4, shuffle=True)

    if args.opt == 'sgd':
        optS = optim.SGD(S.parameters(), lr=args.lr_clone, momentum=0.5, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, args.epochs)
    else:
        optS = optim.Adam(S.parameters(), lr=args.lr_clone, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, args.epochs)

    #inintial model S
    # for epoch in range(1, args.epochs + 1):
    #     train_loss, train_acc = attack_utils.train_soft_epoch(S, args.device, sur_dataset_loader, optS)
    #     test_loss, test_acc = test(S, args.device, test_loader)
    #     tar_acc_fraction = test_acc / tar_acc
    #     print('Epoch: {}; Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f} ({:.2f}x)%\n'.format(epoch,train_loss, train_acc,
    #                                                                                            test_acc, tar_acc_fraction))

    rng = np.random.RandomState()
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    #optS = optim.Adam(S.parameters(), lr=args.lr_clone)

    # Compute number of actions
    #print('y_sub.shape',y_sub.shape)
    if len(y_sub.shape) == 2:
        nb_actions = len(np.unique(np.argmax(y_sub, axis=1)))
        #print('nb_actions1',nb_actions)
    elif len(y_sub.shape) == 1:
        nb_actions = len(np.unique(y_sub))
        #print('nb_actions2',nb_actions)
    else:
        raise ValueError("Target values `y` has a wrong shape.")


    #print("Images shape:", images_np.shape)
    #print("Labels shape:", labels_np.shape)

    h_func = np.zeros(nb_actions)
    learning_rate = np.zeros(nb_actions)
    probs = np.ones(nb_actions) / nb_actions
    reward_avg = np.zeros(3)
    reward_var = np.zeros(3)
    y_avg = np.zeros(num_classes)
    #action = np.random.choice(np.arange(0, nb_actions), p=probs)
    avg_reward = 0.0

    for aug_round in range(args.aug_rounds1):
        # model training
        # Indices to shuffle training set
        index_shuf = list(range(len(X_sub)))
        rng.shuffle(index_shuf)
        if aug_round < args.aug_rounds1 - 1:
            print("[{}] select training data.".format(aug_round+1))
            # Perform adaptive sample selection
            for times in range(args.adaptive_time):
                #print("[{}] samples .".format(times))
                action = np.random.choice(np.arange(0, nb_actions), p=probs)
                #print('X_sub:',X_sub.shape)

                X_sub1= np.array(sample_data(X_sub, y_sub, action)) #从X_sub和y_sub中每一次只选择一个样本

                #print(X_sub1.shape)
                X_sub1=X_sub1.reshape((1,3,32,32))
                y_sub1=get_labels(X_sub1,S)

                # #update S in each selection
                # Sur_inter = [(a, b) for a, b in zip(X_sub1, y_sub1)]
                # sur_dataset_loader_inter = torch.myutils.data.DataLoader(Sur_inter, batch_size=args.batch_size, num_workers=4,
                #                                                  shuffle=True)
                # train_loss, train_acc = attack_utils.train_soft_epoch(S, args.device, sur_dataset_loader_inter, optS)
                #
                # test_loss, test_acc = test(S, args.device, test_loader)
                # tar_acc_fraction = test_acc / tar_acc
                # print('Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f} ({:.2f}x)%\n'.format(train_loss, train_acc,test_acc,tar_acc_fraction))


                if times==0:
                    selected_x=X_sub1
                else:
                    selected_x = np.concatenate((selected_x, X_sub1),axis=0)
                #print(selected_x.shape)

                selected_y = get_labels(X_sub1, T)

                # # training surrogate model 更新一次
                # Sout = S(to_var(torch.from_numpy(X_sub1)))
                # Sout = F.softmax(Sout, dim=1)
                # lossS = criterion(Sout, to_var(torch.from_numpy(selected_y)))
                # optS.zero_grad()
                # lossS.backward()
                # optS.step()

                selected_y_S = get_labels(X_sub1, S)

                #computing reward
                reward,y_avg,reward_avg, reward_var = reward_all(X_sub1, selected_y, selected_y_S, y_avg, times, num_classes, reward_avg, reward_var)
                #print('reward',reward)
                avg_reward = avg_reward + (1.0 / (times+1)) * (reward - avg_reward)

                # Update learning rate
                learning_rate[action] += 1

                # Update H function
                for i_action in range(nb_actions):
                    if i_action != action:
                        h_func[i_action] = (
                                h_func[i_action] - 1.0 / learning_rate[action] * (reward - avg_reward) * probs[i_action]
                        )
                    else:
                        h_func[i_action] = h_func[i_action] + 1.0 / learning_rate[action] * (reward - avg_reward) * (
                                1 - probs[i_action]
                        )

                # Update probs
                aux_exp = np.exp(h_func)
                #print('aux_exp',aux_exp)
                probs = aux_exp / np.sum(aux_exp)


            #constructing  args.adaptive_time samples
            X_sub2 = selected_x
            #print('X_sub for training:',X_sub2.shape)
            y_sub2 = get_labels(X_sub2, T)

        Sur2 = [(a, b) for a, b in zip(X_sub2, y_sub2)]
        sur_dataset_loader2 = torch.utils.data.DataLoader(Sur2, batch_size=args.batch_size, num_workers=4,
                                                         shuffle=True)
        print("[{}] select training data.".format(aug_round+1))
        print('Margin of samples:', margin(X_sub2,y_sub2))
        for epoch in range(1, args.epochs + 1):
            S.train()
            train_loss, train_acc = attack_utils.train_soft_epoch(S, args.device, sur_dataset_loader2, optS)
            #schS.step() #更新学习率
            test_loss, test_acc = test(S, args.device, test_loader)
            tar_acc_fraction = test_acc / tar_acc
            print(
                'Epoch: {} Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f} ({:.2f}x)%\n'.format(epoch, train_loss,
                                                                                                 train_acc,
                                                                                                 test_acc,
                                                                                                     tar_acc_fraction))
        # for epoch in range(args.epochs):
        #     nb_batches = int(np.ceil(float(len(X_sub)) / args.batch_size))
        #     assert nb_batches * args.batch_size >= len(X_sub)
        #
        #     for batch in range(nb_batches):
        #         start, end = batch_indices(batch, len(X_sub), args.batch_size)
        #         x = X_sub[index_shuf[start:end]]
        #         #print('x',x.shape)
        #         y = y_sub[index_shuf[start:end]]
        #         #print('y',y.shape)
        #         Sout = S(to_var(torch.from_numpy(x)))
        #         Sout = F.softmax(Sout, dim=1)
        #         lossS = criterion(Sout, to_var(torch.from_numpy(y)))
        #         optS.zero_grad()
        #         lossS.backward()
        #         optS.step()
        #     test_loss, test_acc = test(S, args.device, test_loader)
        #     print('test_loss:',test_loss,'test_acc',test_acc)


        # If we are not in the last substitute training iteration, augment dataset
        #selected_x = np.zeros(1,3,32,32)

        #print(nb_actions)

        print('Aug Round {} Clone Accuracy: {:.2f}({:.2f})x'.format(aug_round, test_acc, test_acc / tar_acc))



def sample_data(x: np.ndarray, y: np.ndarray, action: int) -> np.ndarray:
    """
    Sample data with a specific action.

    :param x: An array with the source input to the victim classifier.
    :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
              (nb_samples,).
    :param action: The action index returned from the action sampling.
    :return: An array with one input to the victim classifier.
    """
    if len(y.shape) == 2:
        y_index = np.argmax(y, axis=1)
    else:
        y_index = y

    x_index = x[y_index == action]
    #print('The data is',action)
    #print('len(x_index)',len(x_index))

    rnd_idx = np.random.choice(len(x_index))
    return x_index[rnd_idx]

def reward_all(x_sub,y_sub,y_sub_S, y_avg, aug_round, nb_classes,reward_avg, reward_var):
    # Div+loss+Cert

    #Cert
    reward_cert=0
    largests = np.partition(y_sub.flatten(), -2)[-2:]
    reward_cert= largests[1] - largests[0]

    #Div
    y_avg = y_avg + (1.0 / (aug_round+1)) * (y_sub[0] - y_avg)
    reward_div = 0
    for k in range(nb_classes):
        reward_div += np.maximum(0, y_sub[0][k] - y_avg[k])

    #Loss
    aux_exp = np.exp(y_sub[0])
    probs_output = aux_exp / np.sum(aux_exp)

    aux_exp = np.exp(y_sub_S[0])
    probs_hat = aux_exp / np.sum(aux_exp)
    reward_loss = 0
    for k in range(nb_classes):
        reward_loss  += -probs_output[k] * np.log(probs_hat[k])

    #all
    #print(reward_loss,reward_div,reward_cert)

    reward = [reward_cert, reward_div, reward_loss]

    #print('reward_avg_before', reward_avg)
    reward_avg = reward_avg + (1.0 / nb_classes) * (reward - reward_avg)
    #print('reward_avg',reward_avg)

    reward_var = reward_var + (1.0 / nb_classes) * ((reward -reward_avg) ** 2 - reward_var)
    #print('reward_var',reward_var)
    # Normalize rewards
    if nb_classes > 1:
        reward = (reward - reward_avg) / np.sqrt(reward_var)
    else:
        reward = [max(min(r, 1), 0) for r in reward]

    #print('reward',reward)

    return np.nanmean(reward),y_avg,reward_avg,reward_var

def margin(X_sub,y_sub):
    # 返回平均margin值
    print('y_sub.shape_before:',y_sub.shape)
    margin_values = []
    for i in range(X_sub.shape[0]):
        # 找到样本所在的类别
        #print(y_sub[i])
        y_pred = np.argmax(y_sub[i])
        #print(y_pred)
        # 计算margin值
        margin_i = y_sub[i][y_pred] - np.max(np.delete(y_sub[i], y_pred))
        margin_values.append(margin_i)
    return np.mean(margin_values)