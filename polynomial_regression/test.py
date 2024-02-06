import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

def my_function(x):
    y = 12 * x**3 - 6 * x**2 -34 * x + 57
    return y

class MyDataset(Dataset):
    def __init__(self, num_data, min_x=-1000., max_x=1000.):
        super(MyDataset, self).__init__()

        self.data_samples = self.get_data_samples(num_data, min_x, max_x)

    def get_data_samples(self, num_data, min_x, max_x):
        x = torch.rand(num_data) * (max_x - min_x) + min_x
        y = my_function(x)

        data_samples = []
        for i in range(num_data):
            data_samples.append({'x': x[i],
                                 'y': y[i]})
        return data_samples

    def __getitem__(self, index):
        item = self.data_samples[index]
        return item

    def __len__(self):
        return len(self.data_samples)

    def collate_fn(self, batch):
        x_batch = []
        y_batch = []
        for cur_batch in batch:
            x_batch.append(cur_batch['x'])
            y_batch.append(cur_batch['y'])

        batch_out = {'x': torch.tensor(x_batch).reshape(-1, 1),
                     'y': torch.tensor(y_batch).reshape(-1, 1)}
        return batch_out

if __name__ == "__main__":

    # device = torch.device('cuda:0')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = torch.load('model/best_model.pth')
    model.to(device)
    dataset = MyDataset(num_data=10000)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                            collate_fn=dataset.collate_fn)

    model.eval()
    for batch in tqdm(dataloader):
        x = batch['x'].to(device)
        y_gt = batch['y'].to(device)
        y_pred = model(x)

        print("Input: ", x[0,0].item())
        print("Predict: ", y_pred[0, 0].item())
        print("Ground Truth: ", y_gt[0, 0].item())



