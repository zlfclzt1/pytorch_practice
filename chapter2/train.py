import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


from utils.pooling import MaxPooling2d
from utils.conv import Conv2d
from utils.normalization import BatchNorm2D
from utils.pooling import AvgPooling2d
from utils.linear import Linear
from utils.soft_max import SoftMax

# 定义汽车图片文件夹和非汽车背景图文件夹的路径
CAR_DIR = "car_classification/car"
BK_DIR = "car_classification/background"
WORK_DIR = "test_train"

class MyDataset(Dataset):
    def __init__(self, car_dir: str, bk_dir: str, is_train: bool):
        super(MyDataset, self).__init__()
        # 列举汽车图片文件夹和非汽车背景图文件夹内所有文件的路径，并打上标签
        car_files = [{"file": os.path.join(car_dir, file), "is_car": 1}
                     for file in os.listdir(car_dir)]
        bk_files = [{"file": os.path.join(bk_dir, file), "is_car": 0}
                    for file in os.listdir(bk_dir)]

        self.data_samples = car_files + bk_files
        self.is_train = is_train

        # # 使用Albumentations定义数据增广流程
        # self.transform = A.Compose([A.RandomBrightnessContrast(p=0.5),
        #                             A.ISONoise(p=0.5),
        #                             A.RandomSnow(p=0.2)])
    def __getitem__(self, index):
        image = np.array(Image.open(self.data_samples[index]["file"])).astype(float) / 255.0
        label = self.data_samples[index]["is_car"]

        data_info = {
            "image": image,
            "is_car": label
        }
        return data_info

    def __len__(self):
        return len(self.data_samples)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.max_pool = MaxPooling2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2D(32)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2D(64)
        self.conv3 = Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = BatchNorm2D(128)

        self.ave_pool = AvgPooling2d(kernel_size=8, stride=1, padding=0)
        self.linear = Linear(in_features=128, out_features=2)
        self.softmax = SoftMax()
    def forward(self, x):
        # 对于输入图片x进行前向运算，x的维度为32x3x64x64
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))
        # x维度现为32x32x32x32
        x = self.max_pool(self.relu(self.bn2(self.conv2(x))))
        # x维度现为32x64x16x16
        x = self.max_pool(self.relu(self.bn3(self.conv3(x))))
        # x维度现为32x128x8x8
        x = self.ave_pool(x).squeeze()
        # x维度现为32x128
        x = self.linear(x)
        # x维度现为32x2
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    # 定义训练集的DataSet和DataLoader实例
    dataset = MyDataset(car_dir=CAR_DIR,
                        bk_dir=BK_DIR,
                        is_train=True)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,
                            num_workers=12, drop_last=True)

    # 获得当前最佳设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.05)

    # 定义交叉熵损失函数
    celoss = nn.CrossEntropyLoss()
    best_loss = np.inf

    for epoch in range(200):
        # 将模型设置为训练模式
        model.train()
        # 将PyTorch设置为计算梯度的模式
        torch.set_grad_enabled(True)
        for batch in tqdm(dataloader):
            # 每一次迭代开始前，将梯度清零
            optimizer.zero_grad()
            # 将数据转换到模型能接受的格式
            image = batch["image"].permute((0, 3, 1, 2)).float().to(device)
            label = batch["is_car"].long().to(device)

            # 进行前向运算
            out = model(image)
            # 计算交叉熵损失
            loss = celoss(out, label)
            # 如果损失值比之前的最小值更小，保存模型
            if best_loss > loss.item():
                best_loss = loss.item()
                torch.save(model, os.path.join(WORK_DIR, "best_model.pth"))
                print(f"Best loss updated {loss.item()}")
            # 从损失值开始后向传播梯度
            loss.backward()
            # 使用优化器优化模型
            optimizer.step()
        torch.save(model, os.path.join(WORK_DIR, f"model_{epoch}.pth"))
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
