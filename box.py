import numpy as np
import torch
import math
from typing import Literal
from .utils import add_batch
from .dtype import astype

def boxes_to_batch(arr:np.ndarray|list|torch.Tensor)->np.ndarray|torch.Tensor:
    
    a = arr 
    if isinstance(arr, list):
        a = np.asarray(arr)

    if a.ndim == 1:
        a = add_batch(a)
    
    return a

def _to_calculate_dtype(boxes:np.ndarray|list|torch.Tensor, batch:bool=True)->np.ndarray|torch.Tensor:
    
    b = boxes 
    if isinstance(boxes, list) :
        b = np.asarray(boxes, dtype=np.float32)
    b = astype(b, 'float')

    if batch:
        return boxes_to_batch(b)
    
    return b

def scale_box(boxes:np.ndarray|list|torch.Tensor, imgsize:np.ndarray|list|tuple, direction:Literal['normalize', 'back']='normalize')->np.ndarray|torch.Tensor:
    
    """
    Normalize or scale-back bounding boxes for a single image.
    
    Arg:
    -------
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

    Return:
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
    
    b = _to_calculate_dtype(boxes=boxes)
    n = imgsize if isinstance(imgsize, np.ndarray) else np.asarray(imgsize)
    n = np.tile(n, 2).astype(np.float32)
    
    if isinstance(b, torch.Tensor):
        n = torch.from_numpy(n).to(device=b.device)
    
    match direction:
        case 'normalize':
            return b/n
        case 'back':
            return b*n

def shift_box(boxes:np.ndarray|list|torch.Tensor, delta_xy:tuple, format:Literal['xyxy', 'xywh']='xyxy')->np.ndarray|torch.Tensor:
    
    b = _to_calculate_dtype(boxes=boxes)
    delta = np.asarray(delta_xy).astype(np.float32)
    
    if format == 'xyxy':
        delta = np.tile(delta, 2)

    if isinstance(b, torch.Tensor):
        delta = torch.from_numpy(delta).to(device=b.device)
    
    match format: 
        case 'xyxy':
            b += delta
        case 'xywh':
            b[:, :2] += delta
    
    return b
    
def xyxy2xywh(xyxy:np.ndarray|list|torch.Tensor) -> np.ndarray|torch.Tensor:

    xyxy_arr = _to_calculate_dtype(xyxy)
    xywh_arr = None 
    if isinstance(xyxy_arr , np.ndarray):
        xywh_arr = np.zeros_like(xyxy_arr)
    elif isinstance(xyxy_arr, torch.Tensor):
        xywh_arr = torch.zeros_like(xyxy_arr)
    else:
        raise NotImplementedError()
    
    xywh_arr[:, 0] = (xyxy_arr[:, 0] + xyxy_arr[:, 2])/2 #cx
    xywh_arr[:, 1] = (xyxy_arr[:, 1] + xyxy_arr[:, 3])/2 #cy
    xywh_arr[:, 2] = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) #w
    xywh_arr[:, 3] = (xyxy_arr[:, 3] - xyxy_arr[:, 1]) #h
    return xywh_arr

def xywh2xyxy(xywh:np.ndarray|list|torch.Tensor) -> np.ndarray|torch.Tensor:
    
    xywh_arr = _to_calculate_dtype(xywh)
    xyxy_arr = None 
    if isinstance(xywh_arr, np.ndarray):
        xyxy_arr = np.zeros_like(xywh_arr)
    elif isinstance(xywh_arr, torch.Tensor):
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
    return [[quantize(c, i) for i,c in enumerate(b)] for b in boxes_to_batch(xyxy_)]

def box_geo_scale(xywh:torch.Tensor|np.ndarray, scale:float=1) -> torch.Tensor|np.ndarray:
    xywh_ = boxes_to_batch(xywh)
    diag = (xywh_[:, 2]*scale)**2 + (xywh_[:, 3]*scale)**2
    if isinstance(diag, np.ndarray):
        return np.sqrt(diag)
    elif isinstance(diag, torch.Tensor):
        return torch.sqrt(diag)
    
