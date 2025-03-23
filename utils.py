import numpy as np
from typing import Literal
import torch
try:
    import jax.numpy as jnp
    import jaxlib
except:
    pass


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
    },
    'jnp':{
        'int':jnp.int32,
        'long':jnp.int16,
        'float':jnp.float32,
        'double':jnp.float64,
        'bool':jnp.bool
    }
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

