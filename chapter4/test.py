import torch

from utils.dataset import MunichDataset


DATA_PATH = './data'



if __name__ == "__main__":
    dataset = MunichDataset(DATA_PATH)
    print(1)