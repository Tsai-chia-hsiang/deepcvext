import numpy as np
from typing import Literal, Optional
import torch
import cv2


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
try:
    import jax.numpy as jnp
    import jaxlib
    FRAMEWORK_DTYPE_MAP['jnp']={
        'int':jnp.int32,
        'long':jnp.int16,
        'float':jnp.float32,
        'double':jnp.float64,
        'bool':jnp.bool
    }
    def as_type(arr:np.ndarray|torch.Tensor|jnp.ndarray, dtype:Literal['int','long', 'float', 'double', 'bool'])->np.ndarray|torch.Tensor|jnp.ndarray:

        match type(arr):
            case np.ndarray:
                return arr.astype(dtype=FRAMEWORK_DTYPE_MAP['np'][dtype])
            case torch.Tensor:
                return arr.to(dtype=FRAMEWORK_DTYPE_MAP['torch'][dtype])
            case jnp.ndarray:
                return arr.astype(dtype=FRAMEWORK_DTYPE_MAP['jnp'][dtype])
            case jaxlib.xla_extension.ArrayImpl:
                return arr.astype(dtype=FRAMEWORK_DTYPE_MAP['jnp'][dtype])
            case _:
                raise NotImplementedError(f"{type(arr)} is not support.")

except:
    pass

def as_type(arr:np.ndarray|torch.Tensor, dtype:Literal['int','long', 'float', 'double', 'bool'])->np.ndarray|torch.Tensor:

    match type(arr):
        case np.ndarray:
            return arr.astype(dtype=FRAMEWORK_DTYPE_MAP['np'][dtype])
        case torch.Tensor:
            return arr.to(dtype=FRAMEWORK_DTYPE_MAP['torch'][dtype])
        case _:
            raise NotImplementedError(f"{type(arr)} is not support.")

# [src channel][target channel]
_cvtflag = (
    (None, None, None, None), 
    (None, None, None, cv2.COLOR_GRAY2BGR, cv2.COLOR_GRAY2BGRA), # src: gray
    (None, None, None,None,None),
    (None, cv2.COLOR_BGR2GRAY, None, None, cv2.COLOR_BGR2BGRA), # src: bgr
    (None, cv2.COLOR_BGRA2GRAY, None, cv2.COLOR_BGRA2BGR, None) # src: bgra
)

def color_cvt(img:np.ndarray, to_channel:Optional[int]=None):
    global _cvtflag
    i = img.copy().astype(np.uint8)
    if img.ndim == 2:
        # binary img, first expand the channel dimension
        i = np.expand_dims(i, axis=-1)
    if to_channel is None:
        return i
    
    h,w,c = i.shape

    flag = _cvtflag[c][to_channel]
    if flag is not None:
        i = cv2.cvtColor(i, flag)
    return i

