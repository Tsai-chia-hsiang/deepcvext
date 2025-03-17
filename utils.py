import numpy as np
import cv2
from typing import Optional
import torch

def to_batch(arr:np.ndarray|list|torch.Tensor)->np.ndarray|torch.Tensor:
    a = arr 
    if isinstance(arr, list):
        a = np.asarray(arr)
    if a.ndim == 1:
        if isinstance(arr, np.ndarray):
            a = np.expand_dims(a, axis=0)
        elif isinstance(a, torch.Tensor):
            a = a.unsqueeze(0)
        else:
            raise NotImplementedError()
    return a

def find_gathered_places(series, threshold=2.0):
    series = np.array(series)
    
    # Compute Z-score (standardized values)
    mean = np.mean(series)
    std = np.std(series)
    z_scores = (series - mean) / std  # Standardization
    
    # Find indices where the absolute Z-score is above threshold (outliers)
    gathered_indices = np.where(np.abs(z_scores) >= threshold)[0]

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