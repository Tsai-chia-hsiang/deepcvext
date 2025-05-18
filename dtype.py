import numpy as np 
import torch
from .warning_msgs import _UNSUPPORT_TYPE_WARNING_

DTYPE_MAP = {
    'np':{
        'uint8':np.uint8,
        'int':np.int32,
        'long':np.int64,
        'float':np.float32,
        'double':np.float64,
        'bool':np.bool_
    },
    'torch':{
        'uint8':torch.uint8,
        'int':torch.int32,
        'long':torch.long,
        'float':torch.float32,
        'double':torch.float64,
        'bool':torch.bool
    }
}


def isdtype(arr:np.ndarray|torch.Tensor, dtype:str):
    match type(arr):
        case np.ndarray:
            return np.isdtype(arr, DTYPE_MAP['np'][dtype])
        case torch.Tensor:
            return arr.dtype == DTYPE_MAP['torch'][dtype]
        case _:
            raise TypeError(_UNSUPPORT_TYPE_WARNING_(arr))
    
def astype(arr:np.ndarray|torch.Tensor, dtype:str)->np.ndarray|torch.Tensor:
    
    match type(arr):
        case np.ndarray:
            return arr.astype(dtype=DTYPE_MAP['np'][dtype])
        case torch.Tensor:
            return arr.to(dtype=DTYPE_MAP['torch'][dtype])
        case _:
            raise TypeError(_UNSUPPORT_TYPE_WARNING_(arr))