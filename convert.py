import torch
from torchvision import transforms as T
import numpy as np
import cv2

try:
    import jax
    import jax.numpy as jnp
except:
    pass

from typing import Callable, Any, Optional, Literal

__all__ = ["tensor2img", "img2tensor"]


_IMG_NORMALIZE_= lambda x: x/255
_TO_IMG_ = lambda x: np.clip(x, 0, 1)*255

def cvtcolor_to_dl(img:np.ndarray)->np.ndarray:
    if img.ndim == 3 and img.shape[-1] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # gray scale:
        return np.expand_dims(img, axis=-1)  
    
def _to_dl_frame(img:np.ndarray|list[np.ndarray], is_cv2:bool=True, to_batch:bool=True)->np.ndarray:
    if isinstance(img, list):
        i = np.stack([cvtcolor_to_dl(img_i) if is_cv2 else img_i for img_i in img], axis=0)
    elif isinstance(img, np.ndarray):
        assert img.ndim < 4 , f"If wanting batch, using list[np.ndarray] over stacking np.ndarry to 4d array"
        i = cvtcolor_to_dl(img) if is_cv2 else img
    else:
        raise NotImplementedError(f"no {type(img)} such a class support")
    if to_batch:
        i = np.expand_dims(i, axis=0)
    return i.astype(np.float32)

def _img_debatch(img:np.ndarray, to_cv2:bool=True) -> list[np.ndarray]|np.ndarray:
      img = np.clip(img, 0, 255).astype(np.uint8)
      match img.ndim:
        case 4:
            return [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) if to_cv2 else i for i in img]
        case 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if to_cv2 else img
    

def tensor2img(timg:torch.Tensor, scale_back_f:Optional[Callable[[np.ndarray, Any], np.ndarray]]=_TO_IMG_, to_cv2:bool=True, **scale_back_kwargs) -> np.ndarray|list[np.ndarray]:

    """
    Convert a PyTorch RGB tensor image to a NumPy uint8 image.
    
    This function supports both single image and batched images:
    - For a single image (C x H x W), it returns a single NumPy array with dtype uint8.
    - For a batch of images (B x C x H x W), it returns a list of NumPy arrays with dtype uint8.

    Arg:
    --
    - timg (torch.Tensor): 
        A PyTorch tensor representing the image(s). The input should have either:
        - Shape (C x H x W) for a single image.
        - Shape (B x C x H x W) for a batch of images.
    - scale_back_f (Callable, Default is lambda x:x*255): 
        - A function to scale the input tensor's values back to the range [0, 255].
        - If no scaling is needed, set this to `None`. Defaults to `None`.
        - The function should have the signature: 
            - `scale_back_f(timg: torch.Tensor, **kwargs) -> torch.Tensor`
    - to_cv2 (bool, Default True): 
        If `True`, converts the output image(s) to BGR format (for use with OpenCV).
        If `False`, the output image(s) remain in RGB format. Defaults to `True`.
    - **kwargs: 
        Additional parameters to be passed to `scale_back_f`, if specified.

    Return:
    --
    np.ndarray | list[np.ndarray]: 
        - If `timg` is a single image (C x H x W), returns a NumPy array (H x W x C).
        - If `timg` is a batch of images (B x C x H x W), returns a list of NumPy arrays, 
            each with shape (H x W x C).

    Raises:
        ValueError: 
            If the input tensor's dimensions are not 3 (C x H x W) or 4 (B x C x H x W).
    """
    if timg.ndim > 4 or timg.ndim < 3:
        raise ValueError(f"Expected tensor with xpected tensor with ndim=3 or 4, but got ndim={timg.ndim}")

    t0 = timg.detach().cpu()
    # (B)xCxHxW -> (B)xHxWxC
    t0 = t0.permute(1, 2, 0) if t0.ndim == 3 else t0.permute(0,2,3,1)
    img = t0.numpy()
    if scale_back_f is not None:
        img = scale_back_f(img, **scale_back_kwargs)  # Call with kwargs
    return _img_debatch(img=img, to_cv2=to_cv2)

def img2tensor(img:np.ndarray|list[np.ndarray], is_cv2:bool=True, scale_f:Optional[Callable[[torch.Tensor, Any], torch.Tensor]]=_IMG_NORMALIZE_, to_batch:bool=True, **scale_f_kwargs) -> torch.Tensor:
    """
    Convert a COLOR img from np array to pytorch tensor

    Arg
    --
    - img: a numpy array for image
        - note that this function is currently designed for single color img
    - is_cv2 (bool, Default True): wether the image is in OpenCV format : BGR
        - `True`: treate img as BGR
        - `False`: treate img as RGB
    - to_batch (bool, default `False`): wether adding an extra dimension as batch
    - scale_f (Callable, Default None) : 
        - the scale function that apply to the image
        - The function should have the signature: 
            - `scale_f(i: torch.Tensor, **kwargs) -> torch.Tensor`
        - Note that it applies `ToTensor()` transform from torchvision.transform before appling it
            - which means that image will be first divided by 255 then applies `scale_f` 
        - Even passing `None`, it still applies `ToTensor()` by default.
    - **kwargs: 
        Additional parameters to be passed to `scale_f`, if specified.

    Return
    --
    A pytorch tensor for img with shape (CxHxW) if `to_batch` is `False`, (1xCxHxW) otherwise.
    - where C is RGB manner
    """
    def convert_to_torch_tensor(x:np.ndarray)-> torch.Tensor:
        tx = torch.from_numpy(x)
        if tx.ndim == 4:
            # with batch
            tx =  tx.permute(0, 3, 1, 2)
        elif tx.ndim == 3:
            # single image
            tx = tx.permute(2,0,1)
        tx = tx.contiguous()
        return tx
    
    i = _to_dl_frame(img=img, is_cv2=is_cv2, to_batch=to_batch)
    i = convert_to_torch_tensor(i)
    if scale_f is not None :
        i = scale_f(i, **scale_f_kwargs)
    return i


def jnp2img(jimg:jnp.ndarray, scale_back_f:Optional[Callable[[np.ndarray, Any], np.ndarray]]=_TO_IMG_, to_cv2:bool=True, **scale_back_kwargs) -> np.ndarray|list[np.ndarray]:
    img = jnp.array(jimg)
    img = scale_back_f(img, **scale_back_kwargs) if scale_back_f is not None else img
    return _img_debatch(img=img, to_cv2=to_cv2)
    
def img2jnp(img:np.ndarray|list[np.ndarray], is_cv2:bool=True, device:tuple[str, int]=('gpu', 0), scale_f:Callable[[jnp.ndarray, Any], jnp.ndarray]=_IMG_NORMALIZE_, to_batch:bool=True, **scale_f_kwargs) -> jnp.ndarray:
    i = _to_dl_frame(img=img, is_cv2=is_cv2, to_batch=to_batch)
    i = jnp.array(i, device=jax.devices(device[0])[device[1]])
    if scale_f is not None :
        i = scale_f(i, **scale_f_kwargs)
    return i

