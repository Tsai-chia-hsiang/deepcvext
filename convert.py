import torch
from torchvision import transforms as T
import numpy as np
import cv2
from typing import Callable, Any, Optional

__all__ = ["tensor2img", "img2tensor"]

_TO_TENSOR_ = T.ToTensor()


def tensor2img(timg:torch.Tensor, scale_back_f:Optional[Callable[[torch.Tensor, Any], torch.Tensor]]=None, to_cv2:bool=True, **kwargs) -> np.ndarray|list[np.ndarray]:

    """
    Convert a PyTorch RGB tensor image to a NumPy uint8 image.
    
    This function supports both single image and batched images:
    - For a single image (C x H x W), it returns a single NumPy array with dtype uint8.
    - For a batch of images (B x C x H x W), it returns a list of NumPy arrays with dtype uint8.

    Args:
    --
    - timg (torch.Tensor): 
        A PyTorch tensor representing the image(s). The input should have either:
        - Shape (C x H x W) for a single image.
        - Shape (B x C x H x W) for a batch of images.
    - scale_back_f (Callable, optional): 
        - A function to scale the input tensor's values back to the range [0, 255].
        - If no scaling is needed, set this to `None`. Defaults to `None`.
        - The function should have the signature: 
            - `scale_back_f(timg: torch.Tensor, **kwargs) -> torch.Tensor`
    - to_cv2 (bool, Default True): 
        If `True`, converts the output image(s) to BGR format (for use with OpenCV).
        If `False`, the output image(s) remain in RGB format. Defaults to `True`.
    - **kwargs: 
        Additional parameters to be passed to `scale_back_f`, if specified.

    Returns:
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
    t0 = scale_back_f(t0, **kwargs) if scale_back_f is not None else timg
    
    # (B)xCxHxW -> (B)xHxWxC
    t0 = t0.permute(1, 2, 0) if t0.ndim == 3 else t0.permute(0,2,3,1)
    img:np.ndarray = t0.numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    match img.ndim:
        case 4:
            img = [cv2.cvtColor(i, cv2.COLOR_RGB2BGR) if to_cv2 else i for i in img]
        case 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def img2tensor(img:np.ndarray, is_cv2:bool=True, scale_f:Optional[Callable[[torch.Tensor, Any], torch.Tensor]]=None, to_batch:bool=False, **kwargs) -> torch.Tensor:
    """
    Convert a img from np array to pytorch tensor
   
    Args
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
    i = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if is_cv2 else img
    #scale to 0~1
    i = _TO_TENSOR_(i)
    
    if scale_f is not None:
        i = scale_f(i, **kwargs)
    
    if to_batch:
        i = i.unsqueeze(0)

    return i



