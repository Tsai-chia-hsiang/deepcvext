import numpy as np
import torch
import cv2
import math
from typing import Literal, Iterable


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

def _to_calculate_dtype(boxes:np.ndarray|list|torch.Tensor, to_batch:bool=True)->np.ndarray|torch.Tensor:
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
    
    if to_batch:
        return _to_batch(b)
    
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
    return [[quantize(c, i) for i,c in enumerate(b)] for b in _to_batch(xyxy_)]

def box_geo_scale(xywh:torch.Tensor|np.ndarray, scale:float=1) -> torch.Tensor|np.ndarray:
    xywh_ = _to_batch(xywh)
    diag = (xywh_[:, 2]*scale)**2 + (xywh_[:, 3]*scale)**2
    if isinstance(diag, np.ndarray):
        return np.sqrt(diag)
    elif isinstance(diag, torch.Tensor):
        return torch.sqrt(diag)
    
def draw_boxes(img:np.ndarray, xyxy:list[list]|np.ndarray, color:Iterable[tuple[int,int,int]]=None, label:Iterable[str]=None, box_thickness:int=1,text_thickness=2)->None:
    int_xyxy = xyxy2int(xyxy=xyxy)
    c = color
    if c is None:
        c = [(0,0,255) for i in range(len(int_xyxy))]
    else:
        assert len(c) == len(int_xyxy)

    l = label
    if label is not None:
        assert len(l) == len(int_xyxy)
    else:
        l = [None]*len(int_xyxy)
        
    for b,ci,li  in zip(int_xyxy, c, l):    
        cv2.rectangle(img, b[:2], b[2:], color=ci, thickness=box_thickness)
        if li is not None:
            cv2.putText(img, li,(b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ci, thickness=text_thickness)

