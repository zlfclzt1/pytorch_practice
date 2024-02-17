import torch


def iou(box_source, box_target):
    # box_source: 1 * 5 [left_up_x, left_up_y, right_bottom_x, right_bottom_y, score]
    # box_target: N * 5
    left_up_x = torch.max(box_source[:, 0], box_target[:,0])
    left_up_y = torch.max(box_source[:, 1], box_target[:,1])
    right_bottom_x = torch.min(box_source[:, 2], box_target[:, 2])
    right_bottom_y = torch.min(box_source[:, 3], box_target[:, 3])

    inter = torch.clip(right_bottom_x - left_up_x, min=0.) * \
            torch.clip(right_bottom_y - left_up_y, min=0.)

    source_area = (box_source[:, 2] - box_source[:, 0]) * \
                  (box_source[:, 3] - box_source[:, 1])
    target_area = (box_target[:, 2] - box_target[:, 0]) * \
                  (box_target[:, 3] - box_target[:, 1])

    iou = inter / (source_area + target_area - inter)
    return iou

def nms(box_info, iou_threshold=0.5):
    score_index = torch.argsort(box_info[:, -1], descending=True)
    box_info = box_info[score_index]
    result = []

    while(box_info.shape[0] > 0):
        box_source = box_info[0, :].reshape(1, -1)
        box_info = box_info[1:, :]
        iou_result = iou(box_source, box_info)

        result.append(box_source)
        mask = iou_result < iou_threshold
        box_info = box_info[mask, :]

    result = torch.cat(result, dim=0)



if __name__ == "__main__":
    box_source = torch.tensor([[100, 200, 500, 400, 0.8]])
    box_target = torch.tensor([[200, 250, 300, 400, 0.6],
                               [300, 300, 600, 500, 0.7],
                               [600, 200, 800, 400, 0.8],
                               [50, 75, 300, 250, 0.9],
                               [150, 280, 500, 400, 0.5],
                               [110, 400, 580, 600, 0.4],
                               [100, 75, 300, 275, 0.4],])

    # result = iou(box_source, box_target)
    nms = nms(torch.cat([box_source, box_target], dim=0))