import numpy as np
import cv2
from typing import Optional
import torch
from functools import partial
try:
    import jax.numpy as jnp
    import jaxlib
except:
    pass
from .utils import as_type

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



class BinaryMap_Measurer():

    def __init__(self, mean_measure_items:tuple[str]=('precision', "recall", "acc", "f1", 'tn_rate')):
       
        self._overall_metrics = {m:None for m in mean_measure_items}
        self.framework = None
        self.unsqueeze = None
        self.concatenate = None
    

    @staticmethod
    def as_calculation_type(a:np.ndarray|torch.Tensor|jnp.ndarray|dict[str, np.ndarray|torch.Tensor|jnp.ndarray])->np.ndarray|torch.Tensor|jnp.ndarray|dict[str, np.ndarray|torch.Tensor|jnp.ndarray]:
        
        def as_int(arr:np.ndarray|torch.Tensor|jnp.ndarray):
            return as_type(arr, 'int')
    
        return {k: as_int(v) for k,v in a.items()} if isinstance(a, dict) else as_int(a) 
    
    @staticmethod
    def confusion_masks(pred:np.ndarray|torch.Tensor|jnp.ndarray, gt:np.ndarray|torch.Tensor|jnp.ndarray, to_int:bool=False)->dict[str, np.ndarray|torch.Tensor|jnp.ndarray]:
        assert type(pred) == type(gt)
        pred = as_type(arr=pred, dtype='bool')#pred.astype(np.bool)
        gt = as_type(arr=gt, dtype='bool') #gt.astype(np.bool)
        pred_false = ~pred
        gt_false = ~gt
        mask_dict = {
            'tp': (pred & gt).squeeze(),
            'tn': (pred_false & gt_false).squeeze(),
            'fp': (pred & gt_false).squeeze(),
            'fn': (pred_false & gt).squeeze()
        }
        if to_int:
            mask_dict = BinaryMap_Measurer.as_calculation_type(a=mask_dict)
        return mask_dict
    
    @staticmethod
    def pr_f1(confusion_matrix:dict[str, np.ndarray|torch.Tensor|jnp.ndarray]) -> dict[str, np.ndarray|torch.Tensor|jnp.ndarray]:
        
        def safe_div(numerator:np.ndarray|torch.Tensor|jnp.ndarray, denominator:np.ndarray|torch.Tensor|jnp.ndarray)->np.ndarray:
            
            def framework_a_zero(a:np.ndarray|torch.Tensor|jnp.ndarray):
                match type(a):
                    case np.ndarray:
                        return np.array(0, dtype=np.float32)
                    case torch.Tensor:
                        return torch.zeros(0, dtype=torch.float32, device=a.device)
                    case jnp.ndarray:
                        return jnp.zeros(0, dtype=jnp.float32)
            
            return numerator/denominator if denominator > 0 else framework_a_zero(numerator)
        

        confusion_matrix = BinaryMap_Measurer.as_calculation_type(confusion_matrix)
        confusion_matrix = {k: v.sum() for k, v in confusion_matrix.items()}
        M = {
            'acc': safe_div(
                confusion_matrix['tp']+confusion_matrix['tn'], 
                confusion_matrix['tp']+confusion_matrix['tn']+confusion_matrix['fp']+confusion_matrix['fn']
            ),
            'tn_rate':safe_div(
                confusion_matrix['tn'],
                confusion_matrix['tn']+confusion_matrix['fp']
            ) , 
            'precision':safe_div(
                confusion_matrix['tp'],
                confusion_matrix['tp']+confusion_matrix['fp']
            ),
            'recall': safe_div(
                confusion_matrix['tp'],
                confusion_matrix['tp']+confusion_matrix['fn']
            )
        }
        M['f1'] =safe_div(
            2*M['precision']*M['recall'],
            M['precision'] + M['recall']
        )
        return M
 
    def _register_framework(self, a:np.ndarray|torch.Tensor|jnp.ndarray):
        self.framework = type(a)
        match self.framework:
            case np.ndarray:
                self.unsqueeze = partial(np.expand_dims, axis=0)
                self.concatenate = partial(np.concatenate, axis=0)
            case torch.Tensor:
                self.unsqueeze = partial(torch.unsqueeze, dim=0)
                self.concatenate = partial(torch.concatenate, dim=0)
            case jnp.ndarray:
               self.unsqueeze = partial(jnp.expand_dims, axis=0)
               self.concatenate = partial(jnp.concatenate, axis=0)
            case jaxlib.xla_extension.ArrayImpl:
                self.unsqueeze = partial(jnp.expand_dims, axis=0)
                self.concatenate = partial(jnp.concatenate, axis=0)
            case _:
                raise NotImplementedError(f"{type(self.framework)} is not support.")

    def sample_accumulate(self, pred:np.ndarray|torch.Tensor|jnp.ndarray, gt:np.ndarray|torch.Tensor|jnp.ndarray):
        """
        accumulate (logging) the precision, recall, f1, acc, tn-rate 
        for this sample to self._overall_metrics

        If all samples are logged by this function into self._overall_metrics,
        call self.mean_metrics() to get mean-{metrics} dictionary
        """
        if self.framework is None:
            self._register_framework(a=pred)

        assert self.framework == type(pred)
           
        a_sample_metrics = BinaryMap_Measurer.pr_f1(
            confusion_matrix=BinaryMap_Measurer.confusion_masks(
                pred=pred, gt=gt
            )
        )
        
        for k in self._overall_metrics:
            v = self.unsqueeze(a_sample_metrics[k])
            if self._overall_metrics[k] is None:
                self._overall_metrics[k] = v
                continue
            self._overall_metrics[k] = self.concatenate((self._overall_metrics[k], v))
    
    def mean_metrics(self) -> dict[str, float]:
        """
        Using numpy as final protocol to get mean metrics
        - will first using np.asarray() to convert to np.ndarray then using .mean().
        """
        mean_M = {f"mean-{k}":0.0 for k in self._overall_metrics}
        for k, v in self._overall_metrics.items():
            mean_M[f"mean-{k}"] = float(np.asarray(v).mean()) if len(v) else 0.0
        return mean_M
