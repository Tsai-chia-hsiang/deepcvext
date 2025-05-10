import numpy as np 
import torch
from .convert import *

DTYPE_MAP = {
    'np':{
        'int':np.int32,
        'long':np.int64,
        'float':np.float32,
        'double':np.float64,
        'bool':np.bool_
    },
    'torch':{
        'int':torch.int32,
        'long':torch.long,
        'float':torch.float32,
        'double':torch.float64,
        'bool':torch.bool
    }
}

AXIS_MAP = {
    np.ndarray:{
        'axis': 'axis',
        'keepdims': 'keepdims'
    },
    torch.Tensor:{
        'axis': 'dim',
        'keepdims': 'keepdim'
    }
}
