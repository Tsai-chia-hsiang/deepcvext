import numpy as np
from typing import Optional
import torch
import cv2
from .warning_msgs import _UNSUPPORT_TYPE_WARNING_
from .convert import normalize_heatmap
from .dtype import isdtype

def add_batch(arr:np.ndarray|torch.Tensor)->np.ndarray|torch.Tensor:
    match type(arr):
        case np.ndarray:
            return np.expand_dims(arr, 0)
        case torch.Tensor:
            return arr.unsqueeze(0)
        case _:
            raise TypeError(_UNSUPPORT_TYPE_WARNING_(arr))

# [src channel][target channel]
_cvtflag = (
    (None, None, None, None), 
    (None, None, None, cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGRA), # src: gray
    (None, None, None,None,None),
    (None, cv2.COLOR_BGR2GRAY, None, None, cv2.COLOR_BGR2BGRA), # src: bgr
    (None, cv2.COLOR_BGRA2GRAY, None, cv2.COLOR_BGRA2BGR, None) # src: bgra
)

def cvtcolor(img:np.ndarray|list[np.ndarray], to_channel:Optional[int]=None)->np.ndarray|list[np.ndarray]:
    
    def _cvtcolor_single(img0:np.ndarray)->np.ndarray:
    
        if isdtype(img0, 'uint8'):
            i = img0
        elif isdtype(img0, 'float') or isdtype(img0, 'double'):
            i = normalize_heatmap(img)
        elif isdtype(img0, 'long') or isdtype(img0, 'int'):
            assert img0.max() <= 255 and img0.min() >= 0
            i = img0.astype(np.uint8)
                 
        if img0.ndim == 2:
            # binary img, first expand the channel dimension
            i = np.expand_dims(i, axis=-1)
        if to_channel is None:
            return i
        
        h,w,c = i.shape

        flag = _cvtflag[c][to_channel]
        if flag is not None:
            i = cv2.cvtColor(i, flag)
        return i

    if isinstance(img, (list, tuple)):
        return [_cvtcolor_single(img0=i) for i in img]
    elif isinstance(img, np.ndarray):
        return _cvtcolor_single(img0=img)
    else:
        raise TypeError(_UNSUPPORT_TYPE_WARNING_(x=img)+"for cvtcolor")


