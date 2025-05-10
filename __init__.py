import numpy as np 
import torch
from .convert import *

FRAMEWORK_DTYPE_MAP = {
    'np':{
        'int':np.int32,
        'long':np.int64,
        'float':np.float32,
        'double':np.float64,
        'bool':np.bool
    },
    'torch':{
        'int':torch.int32,
        'long':torch.long,
        'float':torch.float32,
        'double':torch.float64,
        'bool':torch.bool
    }
}

