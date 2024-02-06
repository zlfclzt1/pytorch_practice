import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

def my_function(x):
    y = 12 * x**3 - 6 * x**2 -34 * x + 57
    # y = 6 * x**2 - 34 * x + 57
    return y

class MyModel(nn.Module):
    def __init__(self, degree=3):
        super(MyModel, self).__init__()

        self.relu = nn.ReLU()

        # self.linear1 = nn.Linear(degree, 16)
        # self.linear2 = nn.Linear(16, 256)
        # self.linear3 = nn.Linear(256, 256)
        # self.linear4 = nn.Linear(256, 1)
        self.linear1 = nn.Linear(degree, 1)

    def forward(self, input):
        # x = self.relu(self.linear1(input))
        # x = self.relu(self.linear2(x))
        # x = self.relu(self.linear3(x))
        # x = self.linear4(x)
        x = self.linear1(input)
        return x

class MyDataset(Dataset):
    def __init__(self, num_data, degree=3, min_x=-1000., max_x=1000.):
        super(MyDataset, self).__init__()

        self.degree = degree
        self.min_x = min_x
        self.max_x = max_x
        # self.data_samples = self.get_data_samples(num_data, degree, min_x, max_x)
        self.data_samples = [[] for _ in range(num_data)]

    def get_data_samples(self, num_data, degree, min_x, max_x):
        x = torch.rand(num_data) * (max_x - min_x) + min_x
        y = my_function(x)

        data_samples = []
        for i in range(num_data):

            data_samples.append()
        return data_samples

    def __getitem__(self, index):
        x = torch.rand(1) * (self.max_x - self.min_x) + self.min_x
        y = my_function(x)

        cur_x = torch.tensor([x ** (j + 1) for j in range(self.degree)]).reshape(1, -1)
        item = {'x': cur_x,
                'y': y.reshape(-1, 1)}
        return item

    def __len__(self):
        return len(self.data_samples)

    def collate_fn(self, batch):
        x_batch = []
        y_batch = []
        for cur_batch in batch:
            x_batch.append(cur_batch['x'])
            y_batch.append(cur_batch['y'])

        batch_out = {'x': torch.cat(x_batch, dim=0),
                     'y': torch.cat(y_batch, dim=0)}
        return batch_out

if __name__ == "__main__":
    # device = torch.device('cuda:0')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MyModel()
    model.to(device)
    dataset = MyDataset(num_data=100000, degree=3)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8,
                            collate_fn=dataset.collate_fn, drop_last=True)

    optim = optim.Adam(model.parameters(), lr=0.2, weight_decay=0.1)
    mse_loss = nn.MSELoss()
    best_loss = 1e28

    max_epoch = 100
    for epoch in range(max_epoch):
        model.train()
        torch.set_grad_enabled(True)
        for batch in tqdm(dataloader):
            optim.zero_grad()

            x = batch['x'].to(device)
            y_gt = batch['y'].to(device)
            y_pred = model(x)
            loss = mse_loss(y_gt, y_pred)

            if loss < best_loss:
                best_loss = loss.item()
                torch.save(model, 'model/best_model.pth')
                print(f"Best loss updated {loss.item()}")
            loss.backward()
            optim.step()



