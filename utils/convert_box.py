import torch


def xywh2xyxy(inp):
    '''
    Преобразование формата ограничивающих рамок. (x, y, w, h) -> (х, у, х, у)
    Parameters:
        inp: torch. Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
            Формат ограничивающих рамок (x, y, w, h).
    Returns:
        out: torch. Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
            Формат ограничивающих рамок (х, у, х, у).
    '''

    out = torch.empty_like(inp)

    xy = inp[..., :2]  # координаты х и у центра ограничивающей рамки.
    wh = inp[..., 2:] / 2  # половина высоты и ширины ограничивающей рамки.

    out[..., :2] = xy - wh  # координаты х и у левого верхнего угла.
    out[..., 2:] = xy + wh  # координаты х и у правого нижнего угла.
    return out


def xyxy2xywh(inp):
    '''
    Преобразование формата ограничивающих рамок. (х, у, х, y) -> (x, y, w, h)
    Parameters:
        inp: torch. Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
            Формат ограничивающих рамок (х, у, х, у) -
    Returns:
        out: torch. Tensor, size (batch_size, num_boxes, 4) или (num_boxes, 4).
        Формат ограничивающих рамок (x, y, w, h).
    '''
    out = torch.empty_like(inp)

    # координата х центра ограничивающей рамки.
    out[..., 0] = (inp[..., 0] + inp[..., 2]) / 2
    # координата у центра ограничивающей рамки.
    out[..., 1] = (inp[..., 1] + inp[..., 3]) / 2

    # ширина ограничивающей рамки.
    out[..., 2] = inp[..., 2] - inp[..., 0]
    # высота ограничивающей рамки.
    out[..., 3] = inp[..., 3] - inp[..., 1]
    return out
