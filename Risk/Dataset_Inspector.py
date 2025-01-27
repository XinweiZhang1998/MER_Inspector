# 定义选择数据集的函数，包括acc_victim
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations, product


#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def create_dataset_with_acc(fidelity, rc, acc_victim):
    # 计算所有可能的数据组合
    pairs = list(combinations(range(len(fidelity)), 2))
    x = []
    y = []

    for i, j in pairs:
        for a, b in [(i, j), (j, i)]:
            fidelity_diff = fidelity[a] - fidelity[b]
            label = 1 if fidelity_diff > 0 else 0
            #x.append([rc[a], acc_victim[a], rc[b], acc_victim[b],rc[a]-rc[b],acc_victim[a]-acc_victim[b],rc[a]+rc[b],acc_victim[a]+acc_victim[b]])
            #x.append([rc[a], acc_victim[a], rc[b], acc_victim[b],rc[a]-rc[b],acc_victim[a]-acc_victim[b],rc[a]/rc[b],acc_victim[a]/acc_victim[b]])

            #x.append([rc[a], acc_victim[a], rc[b], acc_victim[b],rc[a]-rc[b],acc_victim[a]-acc_victim[b]])
            #x.append([acc_victim[a], acc_victim[b],acc_victim[a]- acc_victim[b]])
            x.append([rc[a], rc[b], rc[a]-rc[b]])

            #without FA
            #x.append([rc[a], acc_victim[a], rc[b], acc_victim[b]])
            #x.append([acc_victim[a], acc_victim[b]])
            #x.append([rc[a], rc[b]])

            #x.append([rc[a], acc_victim[a], rc[b], acc_victim[b]])
            y.append(label)

    return x, y, len(pairs)*2

#组内
def create_intra_group_dataset(fidelity, rc, acc_victim, groups):
    # Intra-group pairs: pairs within each group
    x = []
    y = []
    for group in groups:
        for a, b in combinations(group, 2):
            for pair in [(a, b), (b, a)]:
                fidelity_diff = fidelity[pair[0]] - fidelity[pair[1]]
                label = 1 if fidelity_diff > 0 else 0
                #x.append([rc[pair[0]], acc_victim[pair[0]], rc[pair[1]], acc_victim[pair[1]], rc[pair[0]] - rc[pair[1]], acc_victim[pair[0]] - acc_victim[pair[1]]])
                #x.append([rc[pair[0]], rc[pair[1]], rc[pair[0]] - rc[pair[1]]])
                x.append([ acc_victim[pair[0]], acc_victim[pair[1]], acc_victim[pair[0]] - acc_victim[pair[1]]])
                y.append(label)
    return x, y
#组间
def create_inter_group_dataset(fidelity, rc, acc_victim, groups):
    # Inter-group pairs: pairs between groups
    x = []
    y = []
    for group1, group2 in combinations(groups, 2):
        for a, b in product(group1, group2):
            for pair in [(a, b), (b, a)]:
                fidelity_diff = fidelity[pair[0]] - fidelity[pair[1]]
                label = 1 if fidelity_diff > 0 else 0
                x.append([rc[pair[0]], acc_victim[pair[0]], rc[pair[1]], acc_victim[pair[1]], rc[pair[0]] - rc[pair[1]], acc_victim[pair[0]] - acc_victim[pair[1]]])
                #x.append([rc[pair[0]], rc[pair[1]], rc[pair[0]] - rc[pair[1]]])
                #x.append([ acc_victim[pair[0]], acc_victim[pair[1]], acc_victim[pair[0]] - acc_victim[pair[1]]])
                y.append(label)
    return x, y

groups = [
    list(range(0, 3)),   # 1-3
    list(range(3, 9)),   # 4-9
    list(range(9, 12)),  # 10-12
    list(range(12, 16))  # 13-16
]
# 数据集
datasets = {
    "CIFAR-10": {
        "fidelity": np.array([0.8135, 0.8088, 0.8031, 0.8601, 0.8312, 0.8365, 0.8323, 0.8841, 0.8858, 
                              0.8912, 0.8924, 0.8969, 0.7182, 0.7305, 0.7347, 0.7255]),
        "rc": np.array([0.00353225, 0.00405001, 0.00736076, 0.00350099, 0.0079658, 0.00443948, 
                        0.00973313, 0.00182872, 0.00135096, 0.00186554, 0.00203757, 0.00181087, 
                        0.00179817, 0.00129518, 0.00109987, 0.00131111]),
        "acc_victim": np.array([90.88, 91.92, 92.41, 92.75, 93.27, 93.51, 93.52, 93.90, 94.70, 94.95, 94.84, 95.08, 
                               81.43, 83.26, 83.62, 84.00])
    },
    "STL-10": {
        "fidelity": np.array([0.8135, 0.8088, 0.8031, 0.8601, 0.8312, 0.8365, 0.8323, 0.8841, 0.8858, 
                     0.8912, 0.8924, 0.8969, 0.7182, 0.7305, 0.7347, 0.7255]),
        "rc": np.array([0.8135, 0.8088, 0.8031, 0.8601, 0.8312, 0.8365, 0.8323, 0.8841, 0.8858, 
                     0.8912, 0.8924, 0.8969, 0.7182, 0.7305, 0.7347, 0.7255]),
        "acc_victim": np.array([83.96, 83.79, 84.54, 83.89, 85.28, 85.19, 85.25, 85.61, 87.28, 85.33, 
                           87.4, 86.47, 68.51, 72.55, 73.14, 72.10])
    },
    "FMNIST": {
        "fidelity": np.array([0.9383, 0.9365, 0.9303, 0.9381, 0.9301, 0.9355, 0.931, 0.9427, 0.9449, 
                         0.9432, 0.9482, 0.9437, 0.9339, 0.9379, 0.9324, 0.9252]),
        "rc": np.array([0.00398455, 0.0063199, 0.01225718, 0.00262386, 0.00676822, 0.00481588, 
                   0.01144709, 0.0021598, 0.00141956, 0.00083828, 0.00078157, 0.00075677, 
                   0.00232559, 0.00204182, 0.00167873, 0.00126478]),
        "acc_victim": np.array([93.68, 94.13, 94.07, 94.67, 94.84, 94.85, 95.01, 95.01, 94.84, 
                           95.27, 95.21, 95.17, 92.09, 93.03, 92.61, 92.62])
    },
     "CIFAR-100": {
        "fidelity": np.array([0.5498, 0.5156, 0.4931, 0.5457, 0.5707, 0.5042, 0.505, 0.6221, 0.6579, 
                         0.6633, 0.7266, 0.7353, 0.4164, 0.4321, 0.4064, 0.4294]),
        "rc": np.array([0.0911925, 0.13943514, 0.09815839, 0.06880002, 0.07286134, 0.08139358, 
                   0.0977491, 0.03497861, 0.01825501, 9.14261E-05, 4.60E-05, 7.71E-05, 
                   0.00565036, 0.01246698, 0.02478976, 0.02018757]),
        "acc_victim": np.array([65.76, 66.79, 68.25, 69.22, 69.40, 70.53, 71.03, 73.26, 76.14, 
                           76.13, 77.15, 77.22, 48.35, 50.69, 54.57, 56.68])
    },
    "Celeba": {
        "fidelity": np.array([0.7437, 0.761, 0.7533, 0.765, 0.7801, 0.7783, 0.7703, 0.8104, 0.8324, 
                         0.7938, 0.7963, 0.7913, 0.7282, 0.7485, 0.7402, 0.7523]),
        "rc": np.array([0.09465513, 0.04913824, 0.08989585, 0.12099661, 0.06296331, 0.06557728, 
                   0.07328698, 0.02039603, 0.01143165, 0.07745238, 0.02806437, 0.01891292, 
                   0.00772808, 0.00508702, 0.00413016, 0.00363329]),
        "acc_victim": np.array([76.5, 75.92, 75.98, 77.12, 76.72, 77.28, 77.51, 77.92, 78.22, 
                           77.79, 78.38, 78.23, 75.88, 74.61, 73.92, 73.12])
    }
}

# # 创建MinMaxScaler对象
# scaler = MinMaxScaler()

# # 对每个数据集的rc进行归一化
# for dataset in datasets.values():
#     dataset["rc"] = scaler.fit_transform(dataset["rc"].reshape(-1, 1)).flatten()

# 为每个数据集创建数据
total_pairs = 0
all_x = []
all_y = []
for dataset_name, data in datasets.items():
    print(dataset_name)

    x, y, num_pairs = create_dataset_with_acc(data["fidelity"], data["rc"], data["acc_victim"]) #all
    #x, y = create_intra_group_dataset(data["fidelity"], data["rc"], data["acc_victim"], groups)  #组内
    #x, y = create_inter_group_dataset(data["fidelity"], data["rc"], data["acc_victim"], groups) #组间
    all_x.extend(x)
    all_y.extend(y)
    #total_pairs += num_pairs

all_x=np.array(all_x)
scaler = MinMaxScaler()

# # 对数据进行归一化处理
# all_x = scaler.fit_transform(all_x)

all_y=np.array(all_y)

[m,n]=all_x.shape
print(all_x.shape)
print(all_y.shape)
print(total_pairs)  # 输出所有数据集的总组数

# Convert NumPy arrays to PyTorch tensors
x = torch.tensor(all_x, dtype=torch.float32)
y = torch.tensor(all_y, dtype=torch.float32)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2024)

# 创建归一化器
scaler = MinMaxScaler()
# 仅使用训练集数据来拟合归一化器
scaler.fit(x_train)
# 分别转换训练集和测试集
x_train = scaler.transform(x_train)
print(x_train.shape)
x_test = scaler.transform(x_test)
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)

# Create PyTorch DataLoader for training and testing data
train_data = TensorDataset(x_train, y_train)
test_data = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(n, 64)
        self.relu = nn.ReLU()
        self.fc12 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.drop=nn.Dropout(0.2)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.drop(x)
        x = self.fc12(x)
        x = self.relu(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        #x = self.drop(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Create the model and define the loss function and optimizer
model = SimpleNN()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
#criterion = nn.MSELoss() 

optimizer = optim.Adam(model.parameters(), lr=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000

train_accuracies = []
test_accuracies = []
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))  # Ensure labels have the right shape
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted_train = (outputs >= 0.5).float()  # 根据你的输出调整阈值
        total_train += labels.size(0)
        correct_train += (predicted_train == labels.unsqueeze(1)).sum().item()
    
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")
    # Evaluate the model on the test data
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs >= 0.5).float()
            total_test += labels.size(0)
            correct_test += (predicted == labels.unsqueeze(1)).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')


# 绘制训练和测试准确率图表
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_accuracies, 'go-', label='Training Accuracy')
plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()