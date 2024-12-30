import numpy as np
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

def _to_batch(box:np.ndarray|list)->np.ndarray:
    b = box if isinstance(box, np.ndarray) else np.asarray(box) 
    if b.ndim == 1:
        b = np.expand_dims(b, axis=0)
    return b

def scale_box(boxes:np.ndarray|list, imgsize:np.ndarray|list, direction:Literal['normalize', 'back']='normalize')->np.ndarray:
    
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
    
    b = boxes if isinstance(boxes, np.ndarray) else np.asarray(boxes, dtype=np.float32)
    
    if b.dtype != np.float32:
        b = b.astype(np.float32)
    b = _to_batch(b)
    n = imgsize if isinstance(imgsize, np.ndarray) else np.asarray(imgsize)
    n = np.tile(n, 2).astype(np.float32)
    match direction:
        case 'normalize':
            return b/n
        case 'back':
            return b*n

def xyxy2xywh(xyxy:np.ndarray|list) -> np.ndarray:

    xyxy_arr = _to_batch(xyxy).astype(np.float32)
    xywh_arr = np.zeros_like(xyxy_arr)
    xywh_arr[:, 0] = (xyxy_arr[:, 0] + xyxy_arr[:, 2])/2 #cx
    xywh_arr[:, 1] = (xyxy_arr[:, 1] + xyxy_arr[:, 3])/2 #cy
    xywh_arr[:, 2] = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) #w
    xywh_arr[:, 3] = (xyxy_arr[:, 3] - xyxy_arr[:, 1]) #h
    return xywh_arr

def xywh2xyxy(xywh:np.ndarray|list) -> np.ndarray:
    
    xywh_arr = _to_batch(xywh).astype(np.float32)
    xyxy_arr = np.zeros_like(xywh_arr)
    half_w = xywh_arr[:, 2] /2 
    half_h = xywh_arr[:, 3] /2

    xyxy_arr[:, 0] = xywh_arr[:, 0] - half_w
    xyxy_arr[:, 1] = xywh_arr[:, 1] - half_h
    xyxy_arr[:, 2] = xywh_arr[:, 0] + half_w
    xyxy_arr[:, 3] = xywh_arr[:, 1] + half_h

    return xyxy_arr

def xyxy2int(xyxy:np.ndarray|list) -> list:
    
    def quantize(x, i:int)->int:
        #return int(x)
        if i < 2:
            return math.floor(x)
        else:
            return math.ceil(x)
    
    return [[quantize(c, i) for i,c in enumerate(b)] for b in _to_batch(xyxy)]

"""
def draw_box(img:np.ndarray, xyxy:list, color:tuple[int,int,int]=(0,0,255), thickness=2)->None:
    int_xyxy = xyxy
    if not isinstance(xyxy[0], int):
        int_xyxy = xyxy2int(xyxy=xyxy)[0]
    cv2.rectangle(img, int_xyxy[:2], int_xyxy[2:], color=color, thickness=thickness)
"""

def draw_boxes(img:np.ndarray, xyxy:list[list]|np.ndarray, color:tuple[int,int,int]=(0,0,255), thickness=2)->None:
    int_xyxy = xyxy2int(xyxy=xyxy)
    for b in int_xyxy:    
        cv2.rectangle(img, b[:2], b[2:], color=color, thickness=thickness)
