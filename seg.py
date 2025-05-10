import numpy as np
import cv2
from typing import Optional
import torch
from .utils import astype

def find_gathered_places(series, threshold=2.0):
    series = np.array(series)
    
    # Compute Z-score (standardized values)
    mean = np.mean(series)
    std = np.std(series)
    z_scores = (series - mean) / std  # Standardization
    
    # Find indices where the absolute Z-score is above threshold (outliers)
    gathered_indices = np.where(z_scores >= threshold)[0]

    return gathered_indices # Return indices and values

def connected_components(binary_map:np.ndarray, gthr:Optional[float]=None)->tuple[int, np.ndarray, np.ndarray|None, np.ndarray|None]:
    """
    Returns
    --
    A tuple: 
    - First 2 are the output from `cv2.connectedComponents()`
        - 0. num_labels
        - 1. connected components map 
    - Last 2:
        - if gthr (gathering z-score threshold) is given (i.e. is not `None`):
            - 2. relatively densed components according to the gthr
            - 3. element number for each components
        - Otherwise: `(np.ndarray, None, None)` 
            - 2. None
            - 3. None
    """
    b_map = (binary_map > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(b_map)   
    if gthr is None :
        return num_labels, labels, None, None
    else:
        cluster_sizes = np.bincount(labels.flatten())[1:]
        gather = find_gathered_places(series=cluster_sizes, threshold=gthr)+1

        return num_labels, labels, gather, cluster_sizes 

def binary_dice_score(pred_map:np.ndarray|torch.Tensor, gt_map:np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor:
    """
    Args
    --
    - pred_map: 
        tensor or array in shape `(B) x H x W`
    - gt_map:
        same shape and framework as binary_map
    
    Returns
    --
    dice score : `2*|pred_map ^ gt_map|_0 / (|pred_map|_0 + |gt_map|_0)`
        - It will return as the given type (i.e. if np, then it return np; if torch tensor, it return a tensor with same device as both)
    """
    bool_pred = astype(pred_map > 0, 'int')
    bool_gt = astype(gt_map > 0, 'int')
    sum_axis = (-2, -1)
    u = bool_pred.sum(*sum_axis) + bool_gt.sum(*sum_axis)
    i = (bool_pred*bool_gt).sum(*sum_axis)

    return 2*i/(u+1e-10)
