import torch


def iou(box1, box2):
    '''
    Ограничивающие рамки в формате (x1, y1, x2, y2).

    Parameters:
        box1: torch. Tensor, size (4,)
        box2: torch. Tensor, size (4,)
    Returns:
        iou: torch. Tensor (scalar)
    '''
    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)

    x_left = torch.max(b1x1, b2x1)
    y_top = torch.max(b1y1, b2y1)
    x_right = torch.min(b1x2, b2x2)
    y_bottom = torch.min(b1y2, b2y2)

    if x_right < x_left or y_bottom < y_top:
        return torch. tensor(0, dtype=torch.float)

    w = x_right - x_left
    h = y_bottom - y_top

    inter = w * h
    union = area1 + area2 - inter
    iou = inter / union
    return iou


def box_iou(boxes1, boxes2):
    '''
    Ограничевающие рамки в формате (x1, y1, x2, y2).
    Parameters:
        boxes1: torch. Tensor, size (N, 4)
        boxes2: torch. Tensor, size (M, 4)
    Returns:
        iou: torch. Tensor, size (N, M)
    '''
    # size (N,)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    # size (M,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, : 2])  # size (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # size (N, M, 2)

    wh = (rb - lt).clamp(min=0)  # size (N, M, 2)

    inter = wh[..., 0] * wh[..., 1]  # size (N, M)
    union = area1[:, None] + area2 - inter  # size (N, M)

    iou = inter / union
    return iou
