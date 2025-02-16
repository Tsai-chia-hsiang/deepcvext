import numpy as np
import torch
import cv2
import math
from typing import Literal

__all__ = [
    "scale_box",
    "xyxy2xywh",
    "xywh2xyxy",
    "xyxy_to_int",
    "draw_box"
]

def _to_batch(box:np.ndarray|list|torch.Tensor)->np.ndarray|torch.Tensor:
    b = box 
    if isinstance(box, list):
        b = np.asarray(box)
    if b.ndim == 1:
        if isinstance(box, np.ndarray):
            b = np.expand_dims(b, axis=0)
        elif isinstance(box, torch.Tensor):
            b = b.unsqueeze(0)
        else:
            raise NotImplementedError()
    return b

def scale_box(boxes:np.ndarray|list|torch.Tensor, imgsize:np.ndarray|list|tuple, direction:Literal['normalize', 'back']='normalize')->np.ndarray|torch.Tensor:
    
    """
    Normalize or scale-back bounding boxes for a single image.

    Args:
    ----
    - boxes (np.ndarray | list[list[float]]): 
        The bounding boxes within an image, represented as either a list of lists of floats or a NumPy array.
        Example:
        `
        [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
        `
    - imgsize (np.ndarray | list[int]): 
        The size of the original image as [width, height].
    - direction (Literal['normalize', 'back']): 
        Specifies the operation to perform:
        - `normalize`: Normalizes the bounding box coordinates by dividing them by [width, height].
          - `normalized_boxes = boxes / imgsize`
        - `back`: Scales the normalized coordinates back to the original image size.
          - `original_boxes = boxes * imgsize`

    Returns:
    -------
    np.ndarray: 
        The transformed bounding boxes as a NumPy array with `np.float32` data type, based on the specified direction.

    Raises:
    ------
    KeyError:
        If the `direction` argument is not one of ['normalize', 'back'].
    """

    assert direction  in ['normalize', 'back'], KeyError(
        f"direction `{direction}` you give is not a legal key.\
        This function is for normalizing/scaling back purpose, please chose a task in \
        ['normalize', 'back']"
    )
    
    b = boxes 
    if isinstance(boxes, list) :
        b = np.asarray(boxes, dtype=np.float32)
    
    if isinstance(boxes, np.ndarray):
        if b.dtype != np.float32:
            b = b.astype(np.float32)
    elif isinstance(boxes, torch.Tensor):
        if b.dtype != torch.float32:
            b = b.astype(torch.float32)
    else:
        raise NotImplementedError()

    b = _to_batch(b)
    n = imgsize if isinstance(imgsize, np.ndarray) else np.asarray(imgsize)
    n = np.tile(n, 2).astype(np.float32)
    
    if isinstance(b, torch.Tensor):
        n = torch.from_numpy(n).to(device=b.device)
    
    match direction:
        case 'normalize':
            return b/n
        case 'back':
            return b*n

def xyxy2xywh(xyxy:np.ndarray|list|torch.Tensor) -> np.ndarray|torch.Tensor:

    xyxy_arr = _to_batch(xyxy)
    xywh_arr = None 
    if isinstance(xyxy_arr , np.ndarray):
        xyxy_arr = xyxy_arr.astype(np.float32)
        xywh_arr = np.zeros_like(xyxy_arr)
    elif isinstance(xyxy_arr, torch.Tensor):
        xyxy_arr = xyxy_arr.to(torch.float32)
        xywh_arr = torch.zeros_like(xyxy_arr)
    else:
        raise NotImplementedError()
    
    xywh_arr[:, 0] = (xyxy_arr[:, 0] + xyxy_arr[:, 2])/2 #cx
    xywh_arr[:, 1] = (xyxy_arr[:, 1] + xyxy_arr[:, 3])/2 #cy
    xywh_arr[:, 2] = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) #w
    xywh_arr[:, 3] = (xyxy_arr[:, 3] - xyxy_arr[:, 1]) #h
    return xywh_arr

def xywh2xyxy(xywh:np.ndarray|list|torch.Tensor) -> np.ndarray|torch.Tensor:
    
    xywh_arr = _to_batch(xywh)
    xyxy_arr = None 
    if isinstance(xywh_arr, np.ndarray):
        xywh_arr = xywh_arr.astype(np.float32)
        xyxy_arr = np.zeros_like(xywh_arr)
    elif isinstance(xywh_arr, torch.Tensor):
        xywh_arr = xywh_arr.to(torch.float32)
        xyxy_arr = torch.zeros_like(xywh_arr)
    
    half_w = xywh_arr[:, 2] /2 
    half_h = xywh_arr[:, 3] /2

    xyxy_arr[:, 0] = xywh_arr[:, 0] - half_w
    xyxy_arr[:, 1] = xywh_arr[:, 1] - half_h
    xyxy_arr[:, 2] = xywh_arr[:, 0] + half_w
    xyxy_arr[:, 3] = xywh_arr[:, 1] + half_h

    return xyxy_arr

def xyxy2int(xyxy:np.ndarray|list|torch.Tensor) -> list:
    
    def quantize(x, i:int)->int:
        #return int(x)
        if i < 2:
            return math.floor(x)
        else:
            return math.ceil(x)
    
    xyxy_ = xyxy if not isinstance(xyxy, torch.Tensor) else xyxy.cpu().numpy()
    return [[quantize(c, i) for i,c in enumerate(b)] for b in _to_batch(xyxy_)]


def draw_boxes(img:np.ndarray, xyxy:list[list]|np.ndarray, color:tuple[int,int,int]=(0,0,255), thickness=2)->None:
    int_xyxy = xyxy2int(xyxy=xyxy)
    for b in int_xyxy:    
        cv2.rectangle(img, b[:2], b[2:], color=color, thickness=thickness)
