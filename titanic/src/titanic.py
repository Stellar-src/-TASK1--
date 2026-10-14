import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(42)
df = pd.read_csv("train.csv")
#print(df.isnull().sum())
#print(df.head())
X = df[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
X = pd.get_dummies(X, dummy_na=False)
#print(X)
y = df['Survived']
#print(y)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
#在训练集和数据集分开后再填补数据，避免数据泄露导致测试不真实：
X_train["Age"] = X_train["Age"].fillna(X_train["Age"].median())
X_test["Age"] = X_test["Age"].fillna(X_train["Age"].median())
#print(X.isnull().sum())
#对数据标准化
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
#转成张量
X_train=torch.tensor(X_train.values,dtype=torch.float32)
X_test=torch.tensor(X_test.values,dtype=torch.float32)
y_train=torch.tensor(y_train.values,dtype=torch.float32)
y_test=torch.tensor(y_test.values,dtype=torch.float32)
print(X_train.shape)
#开始构建模型----------------------------------------------------------------这里添加了隐藏层，让模型拥有非线性表达的能力
torch.manual_seed(42)
net = nn.Sequential(
    nn.Linear(7,16),
    nn.ReLU(),

    nn.Linear(16,8),
    nn.ReLU(),

    nn.Linear(8,1),
    nn.Sigmoid()
)
#定义损失函数
loss_fn = nn.BCELoss()
#print(loss)
#定义梯度下降优化器
optimizer=torch.optim.SGD(net.parameters(),lr=1)
loss_history = []
epoch_history = []
train_history = []
test_history = []
for epoch in range(100):
    # 1.预测
    y_hat = net(X_train)
    # 2.计算损失
    loss = loss_fn(y_hat,y_train.reshape(-1,1))
    # 3.计算梯度
    optimizer.zero_grad() #清空梯度，避免梯度累加
    loss.backward()
    # 4.更新w,b
    optimizer.step()
    if epoch % 1 == 0:
        loss_history.append(loss.item())
        epoch_history.append(epoch)
        net.eval()   #切换到测试模式
        with torch.no_grad():
            test_pred = net(X_test)
            train_pred = net(X_train)
        accuracy_train = ((train_pred>0.5).float()==y_train.reshape(-1,1)).float().mean().item()
        accuracy_test = ((test_pred>0.5).float()==y_test.reshape(-1,1)).float().mean().item()
        train_history.append(accuracy_train)
        test_history.append(accuracy_test)
        net.train()  #恢复训练模式

print(net[0].weight)
print(net[0].bias)
#训练集里最终的正确率：
print("训练集里最终的正确率:")
print(train_history[-1])
# 测试集里测试模型
print("测试集最终正确率：")
print(test_history[-1])

#图1
plt.figure(figsize=(10, 6))
plt.plot(epoch_history,loss_history,color='blue', linewidth=2, label='loss')
plt.title(f'train final loss:{loss_history[-1]:.4f}')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.grid(True)
plt.show()


#图2
plt.figure(figsize=(10, 6))
plt.plot(epoch_history,train_history,color='blue', linewidth=2, label='train')
plt.plot(epoch_history,test_history,color='red', linewidth=2, label='tests')
plt.title('Train and Test Accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.grid(True)
plt.show()

#新样本预测功能
raw = input("请输入 Pclass, Is_Female?, Is_Male?, Age, SibSp, Parch, Fare，""用逗号分隔,性别是打1不是打0，例如：2,0,1,66,0,0,10.5\n").strip().split(",")
if len(raw) != 7:
    raise ValueError("请按顺序输入 7 项信息，并用英文逗号分隔")
pclass, female, male, age, sibsp, parch, fare = [item.strip() for item in raw]
example = pd.DataFrame([{"Pclass": int(pclass),"Sex_female": int(female), "Sex_male": int(male), "Age": float(age),"SibSp": int(sibsp),"Parch": int(parch),"Fare": float(fare),}])
example = (example - mean) / std
example=torch.tensor(example.to_numpy(),dtype=torch.float32)
net.eval()
with torch.no_grad():
    output = net(example)
print(f"该乘客预测状态为{['死亡', '幸存'][int(output.item() > 0.5)]}")