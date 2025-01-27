import torch
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from myutils.helpers import test
import torch.nn.functional as F

num_classes_dict = {
   'fashionmnist': 10,
   'cifar10': 10,
   'cifar100': 100,
   'svhn': 10,
   'gtsrb': 43
}


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


def jacobian(model, x, nb_classes=10):
   """
   This function will return a list of PyTorch gradients
   """
   list_derivatives = []
   x_var = to_var(torch.from_numpy(x), requires_grad=True)

   # derivatives for each class
   for class_ind in range(nb_classes):
      x_var_exp = x_var.unsqueeze(0)
      score = model(x_var_exp)[:, class_ind]
      score.backward()
      list_derivatives.append(x_var.grad.data.cpu().numpy())
      x_var.grad.data.zero_()

   return list_derivatives


def pgd_augmentation(model, X_sub_prev, Y_sub, epsilon, alpha, num_iter, nb_classes):
   """
   Create new numpy array for adversary training data with twice as many components on the first dimension.
   """
   X_sub = np.vstack([X_sub_prev, X_sub_prev])
   if Y_sub.ndim == 2:
      # Labels could be a posterior probability distribution. Use argmax as a proxy.
      Y_sub = np.argmax(Y_sub, axis=1)

   # For each input in the previous' substitute training iteration
   offset = len(X_sub_prev)
   for ind, x in enumerate(X_sub_prev):
      x_adv = x.copy()
      y = Y_sub[ind]
      for i in range(num_iter):
         grads = jacobian(model, x_adv, nb_classes)
         grad = grads[y]
         grad_val = np.sign(grad)
         x_adv = x_adv + alpha * grad_val
         x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
         x_adv = np.clip(x_adv, -1, 1)  # ensure valid pixel range
      X_sub[offset + ind] = x_adv
   # Return augmented training data (needs to be labeled afterwards)
   return X_sub


def margin(X_sub, y_sub):
   # 返回平均margin值
   print('y_sub.shape_before:', y_sub.shape)
   margin_values = []
   for i in range(X_sub.shape[0]):
      # 找到样本所在的类别
      # print(y_sub[i])
      y_pred = np.argmax(y_sub[i])
      # print(y_pred)
      # 计算margin值
      margin_i = y_sub[i][y_pred] - np.max(np.delete(y_sub[i], y_pred))
      margin_values.append(margin_i)
   return np.mean(margin_values)


def get_margin(X_sub, y_sub):
   # 返回一个样本margin值
   # print('y_sub.shape_before:', y_sub.shape)
   # print('y_sub:',y_sub)
   y_pred = np.argmax(y_sub)
   margin = y_sub[y_pred] - np.max(np.delete(y_sub, y_pred))
   # print('Margin:', margin)
   return margin


def get_equal_samples(all_data, all_labels, T, num_classes, num):
   samples_per_class = num // num_classes
   selected_samples = []
   for class_label in range(num_classes):
      # Get data and labels for this class
      class_data = all_data[all_labels == class_label]
      class_labels = all_labels[all_labels == class_label]

      # Compute margin and second largest class for each sample in this class
      margins = []
      second_largest_classes = []
      class_y_sub = get_labels(class_data, T)
      for data, y_sub in zip(class_data, class_y_sub):
         margins.append(get_margin(data, y_sub))
         second_largest_classes.append(np.argsort(y_sub)[-2])

      # Convert margins list to a numpy array
      margins = np.array(margins)
      second_largest_classes = np.array(second_largest_classes)
      samples_per_second_class = samples_per_class // (num_classes - 1) + 1
      print("samples_per_second_class:", samples_per_second_class)
      for second_class_label in range(num_classes):
         if second_class_label == class_label:
            continue

         # Get the samples where the second largest class is the current second_class_label
         class_data_sec = class_data[second_largest_classes == second_class_label]
         margins_sec = margins[second_largest_classes == second_class_label]
         print("class_data", class_data_sec.shape)

         if class_data_sec.shape[0] == 0:
            continue

         # Select the samples with the smallest margins in this group
         if class_data_sec.shape[0] > samples_per_second_class:
            smallest_indices = np.argpartition(margins_sec, samples_per_second_class)[:samples_per_second_class]
         else:
            smallest_indices = np.argsort(margins_sec)[:class_data_sec.shape[0]]
            smallest_indices = np.resize(smallest_indices, samples_per_second_class)
            # Repeat the indices if the length is less than samples_per_second_class
            smallest_indices[:samples_per_second_class] = smallest_indices[:class_data_sec.shape[0]]

            # smallest_indices = np.argpartition(margins_sec, class_data_sec.shape[0]-1)[:class_data_sec.shape[0]]
         selected_samples.extend(class_data_sec[smallest_indices])

   return np.array(selected_samples)


def PGD(args, T, S, train_loader, test_loader, tar_acc):
   T.eval()
   S.train()
   train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=128, shuffle=True)
   num_classes = num_classes_dict[args.dataset]
   # Select The large num_seed margin samples
   data_list = []
   labels_list = []

   for data, labels in train_loader:
      data_list.append(data.numpy())
      labels_list.append(labels.numpy())
   all_data = np.concatenate(data_list, axis=0)
   all_labels = np.concatenate(labels_list, axis=0)

   #随机取样本
   # indices = np.random.choice(all_data.shape[0], args.num_seed, replace=False)
   # selected_samples=all_data[indices]

   """
   #直接取margin最大、最小的n个样本
   selected_samples = []
   margins = []
   all_y_sub = get_labels(all_data, T)
   for data, y_sub in zip(all_data, all_y_sub):
       margins.append(get_margin(data, y_sub))
   margins = np.array(margins)
   # margins_order=margins[np.argpartition(margins,args.num_seed)]
   # sorted_margins_order = np.sort(margins_order[:args.num_seed])
   # TEST=np.mean(sorted_margins_order)

   smallest_indices = np.argpartition(margins, args.num_seed)[:args.num_seed]
   selected_samples=all_data[smallest_indices]
   #largest_indices = np.argpartition(margins, -args.num_seed)[-args.num_seed:]
   #selected_samples=all_data[largest_indices]    

   #y_samples=all_y_sub[smallest_indices]
   #y=get_labels(x,T)
   #margin_sample=get_margin(x,y)
   """

   #在每个类中取占比一致的样本
   samples_per_class = args.num_seed // num_classes
   selected_samples = []
   selected_labels =[]
   for class_label in range(num_classes):
       # Get data and labels for this class
       class_data = all_data[all_labels == class_label]
       class_labels = all_labels[all_labels == class_label]

       #delete the misclassied data
       # class_data = np.array(class_data)
       # class_predict= get_labels(class_data, T)
       # class_predict_labels = np.argmax(class_predict, axis=1)
       # mismatch_indices = np.where(class_labels != class_predict_labels)
       # # Remove the mismatched indices from class_predict
       # class_labels = np.delete(class_labels, mismatch_indices)
       # class_data = np.delete(class_data, mismatch_indices, axis=0)

       # Compute margin for each sample in this class
       margins = []
       class_y_sub = get_labels(class_data, T)
       for data, y_sub in zip(class_data, class_y_sub):
           margins.append(get_margin(data, y_sub))

       # Convert margins list to a numpy array
       margins = np.array(margins)

       # smallest margins
       # smallest_indices = np.argpartition(margins, samples_per_class)[:samples_per_class]
       # # #smallest_indices  = np.random.choice(smallest_indices , size=samples_per_class//2, replace=False)
       # selected_samples.extend(class_data[smallest_indices])
       # selected_labels.extend(class_labels[smallest_indices])

       # Random samples
       # indices = np.random.choice(class_data.shape[0], samples_per_class, replace=False)
       # selected_samples.extend(class_data[indices])

       # Largest margins
       largest_indices = np.argpartition(margins, -samples_per_class)[-samples_per_class:]
       #largest_indices  = np.random.choice(largest_indices, size=samples_per_class//2, replace=False)
       selected_samples.extend(class_data[largest_indices])
       selected_labels.extend(class_labels[largest_indices])

   # 在每个类中取占比一致的样本,且位于与各个类的边界上都相等
   #selected_samples = get_equal_samples(all_data, all_labels, T, num_classes, args.num_seed)

   X_sub = np.array(selected_samples)
   y_true = np.array(selected_labels)

   y_sub = get_labels(X_sub, T)
   predicted_labels = np.argmax(y_sub, axis=1)
   num_incorrect = np.sum(predicted_labels != y_true)
   print("incorrect", num_incorrect)
   # Test
   margin_test = []
   for data, y_sub1 in zip(X_sub, y_sub):
      margin_test.append(get_margin(data, y_sub1))
   margin_test = np.array(margin_test)
   sorted_margin = np.sort(margin_test)
   margin_test1 = np.mean(margin_test)
   Test1 = margin(X_sub, y_sub)
   print("Margin_test", Test1)

   #  Label seed data

   """
   data_iter = iter(train_loader)
   X_sub, _ = data_iter.__next__()
   X_sub = X_sub.numpy()
   y_sub = get_labels(X_sub, T)
   """

   rng = np.random.RandomState()
   criterion = torch.nn.KLDivLoss(reduction='batchmean')
   optS = optim.Adam(S.parameters(), lr=args.lr_clone)

   # Train the substitute and augment dataset alternatively
   for aug_round in range(args.aug_rounds):
      # model training
      # Indices to shuffle training set
      index_shuf = list(range(len(X_sub)))
      rng.shuffle(index_shuf)

      print('Margin of samples:', margin(X_sub, y_sub))

      # training
      for epoch in range(args.epochs):
         nb_batches = int(np.ceil(float(len(X_sub)) / args.batch_size))
         assert nb_batches * args.batch_size >= len(X_sub)

         for batch in range(nb_batches):
            start, end = batch_indices(batch, len(X_sub), args.batch_size)
            x = X_sub[index_shuf[start:end]]
            y = y_sub[index_shuf[start:end]]
            Sout = S(to_var(torch.from_numpy(x)))
            Sout = F.softmax(Sout, dim=1)
            lossS = criterion(Sout, to_var(torch.from_numpy(y)))
            optS.zero_grad()
            lossS.backward()
            optS.step()
         test_loss, test_acc = test(S, args.device, test_loader)

      # If we are not in the last substitute training iteration, augment dataset
      if aug_round < args.aug_rounds - 1:
         print("[{}] Augmenting substitute training data.".format(aug_round + 1))
         # Perform the Jacobian augmentation
         print('X_sub.shape_before:', X_sub.shape)
         X_sub = pgd_augmentation(S, X_sub, y_sub, epsilon=0.1, alpha=0.01, num_iter=1, nb_classes=num_classes)
         #X_sub = jacobian_augmentation(S, X_sub, y_sub, nb_classes=num_classes)
         print('X_sub.shape:', X_sub.shape)
         print("Labeling substitute training data.")
         # Label the newly generated synthetic points using the black-box
         y_sub = get_labels(X_sub, T)
      print('Aug Round {} Clone Accuracy: {:.2f}({:.2f})x'.format(aug_round + 1, test_acc, test_acc / tar_acc))



