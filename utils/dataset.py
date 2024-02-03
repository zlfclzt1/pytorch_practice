import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

ClassMap = {
    "vehicle": 0
}

class MunichDataset(Dataset):
    def __init__(self, data_path, interval=1,):
        super(MunichDataset, self).__init__()

        self.data_path = data_path
        self.interval = interval
        self.data_samples = self.get_data_samples(data_path)
        self.data_samples = self.data_samples[::interval]

    def get_data_samples(self, data_path):
        image_folder_path = os.path.join(data_path, "camera")
        anno_folder_path = os.path.join(data_path, "detections")

        data_samples = []
        image_list = os.listdir(image_folder_path)
        image_list.sort()
        for idx, image_file in enumerate(image_list):
            name_str = image_file.split('.')[0]
            data_info = {'image_path': os.path.join(image_folder_path, name_str + ".jpg"),
                         'anno_path': os.path.join(anno_folder_path, name_str + ".json")}
            data_samples.append(data_info)
        return data_samples


    def __getitem__(self, index):
        data = dict()
        image_path = self.data_samples[index]['image_path']
        anno_path = self.data_samples[index]['anno_path']

        image = np.array(Image.open(image_path)).astype(float) / 255.0
        data["image"] = image.copy()
        width = image.shape[1]
        height = image.shape[0]


        box_info = json.load(open(anno_path))
        box_info = box_info["boxes"]
        boxes = []
        class_labels = []
        for box in box_info:
            if box["label"] == 2:
                x1, y1, x2, y2 = box["box"]
                boxes.append(
                    [0.5 * (x1 + x2) / width, 0.5 * (y1 + y2) / height, (x2 - x1) / width, (y2 - y1) / height])
                class_labels.append("vehicle")
        if len(boxes) > 0:
            data["bboxes"] = np.array(boxes)
            data["class_labels"] = class_labels

        if "bboxes" in data and len(data["bboxes"]) > 0:
            data["bboxes"] = np.array(data["bboxes"])
            class_col = torch.Tensor([ClassMap[label]
                                     for label in data["class_labels"]])
            class_col = np.expand_dims(class_col, 1)
            data["bboxes"] = np.concatenate(
                [class_col, data["bboxes"]], axis=1)
        elif "bboxes" in data:
            del data["bboxes"]
        return data

    def __len__(self):
        return len(self.data_samples)

    def collate_fn(self, batch):
        batch_output = dict()
        # list of 3x480x640
        imgs = [torch.Tensor(data["image"]).permute((2, 0, 1))
                for data in batch]
        # torch.Size([32, 3, 480, 640])
        batch_output["image"] = torch.stack(imgs, 0)

        if "seg" in batch[0]:
            # list of torch.Size([1, 480, 640])
            segs = [torch.Tensor(data["seg"]).permute((2, 0, 1))
                    for data in batch]
            # torch.Size([32, 1, 480, 640])
            batch_output["seg"] = torch.stack(segs, 0)

        boxes = []
        for i, data in enumerate(batch):
            if "bboxes" in batch[i]:
                box = torch.Tensor(batch[i]["bboxes"])
                index_col = i * torch.ones((box.shape[0], 1))
                box = torch.cat([index_col, box], dim=1)
                # list of nx6
                boxes.append(box)
        if len(boxes) > 0:
            # Nx6
            batch_output["bboxes"] = torch.cat(boxes, 0)
        return batch_output


if __name__ == "__main__":
    dataset = MunichDataset('../data')
    dataloader = DataLoader(dataset, batch_size=2, num_workers=8,
                            collate_fn=dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        print(1)