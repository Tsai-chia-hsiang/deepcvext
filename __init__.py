import numpy as np 
import torch
from .convert import *


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
