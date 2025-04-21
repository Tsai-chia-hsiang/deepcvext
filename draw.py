import cv2
import numpy as np
from typing import  Iterable, Optional
from .box import xyxy2int

_MAX_CLASSES = 500 
_Ccolors = None

def canvas(imlist:list[np.ndarray], hbar:int=10, wbar:int=10, row:Optional[int]=None, bar_color:tuple[int]=(0,0,0)) -> np.ndarray:

    row = row if row is not None else len(imlist)
    h = np.full((imlist[0].shape[0], hbar, 3), bar_color, dtype=np.uint8)
    w = np.full((wbar, imlist[0].shape[1]*row + hbar*(row-1), 3), bar_color, dtype=np.uint8)

    c = None
    extra = len(imlist) % row 
    imlist_ = imlist
    if extra > 0:
        imlist_ = [i.copy() for i in imlist]
        to_pad = row - extra
        pimg = np.ones_like(imlist[0])*255
        for i in range(to_pad):
            imlist_.append(pimg)
     
    for i in range(0, len(imlist_), row):
        to_cat = []
        imgs = [j for j in imlist_[i:i+row]]
        for ri in range(len(imgs)):
            to_cat.append(imgs[ri])
            if ri< len(imgs) - 1:
                to_cat.append(h)
                
        a_row = cv2.hconcat(to_cat)
        if c is None:
            c = a_row.copy()
        else:
            c = cv2.vconcat([c, w, a_row])

    return c

def draw_boxes(img:np.ndarray, xyxy:list[list]|np.ndarray, color:Iterable[tuple[int,int,int]]=None, label:Iterable[str]=None, box_thickness:int=1,text_thickness=2)->None:
    int_xyxy = xyxy2int(xyxy=xyxy)
    c = color
    if c is None:
        c = [(0,0,255) for i in range(len(int_xyxy))]
    else:
        assert len(c) == len(int_xyxy)

    l = label
    if label is not None:
        assert len(l) == len(int_xyxy)
    else:
        l = [None]*len(int_xyxy)
        
    for b,ci,li  in zip(int_xyxy, c, l):    
        cv2.rectangle(img, b[:2], b[2:], color=ci, thickness=box_thickness)
        if li is not None:
            cv2.putText(img, li,(b[0], b[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ci, thickness=text_thickness)

def draw_segmentation_map(labels:np.ndarray, target_cids:Iterable[int]=None, cid_color_map:dict[int, tuple[int,int,int]]=None) -> np.ndarray:
    global _Ccolors
    color_map = np.zeros((*labels.shape[:2], 3), dtype=np.uint8)  # Color image
    
    t = target_cids if target_cids is not None else np.unique(labels).tolist()
    
    if cid_color_map is None and _Ccolors is None:
        rng = np.random.default_rng(42)
        _Ccolors = rng.integers(low=0, high=256, size=(_MAX_CLASSES, 3), dtype=np.uint8)
    
    for  i in t:
        color_map[labels == i] = _Ccolors[i%_MAX_CLASSES] if cid_color_map is None else cid_color_map[i]
    
    return color_map
