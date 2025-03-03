import cv2
import numpy as np
from pathlib import Path

def canvas(imlist:list[np.ndarray], hbar:int=10, wbar:int=10, row:int=2, save_to:Path=None, need_return:bool=True) -> np.ndarray|None:
    
    h = np.zeros((imlist[0].shape[0], hbar, 3), dtype=np.uint8)
    w = np.zeros((wbar, imlist[0].shape[1]*row + hbar*(row-1), 3), dtype=np.uint8)
    c = None
    extra = len(imlist) % row
    imlist_ = imlist
    if extra > 0:
        imlist_ = [i.copy() for i in imlist]
        to_pad = row - extra
        pimg = np.zeros_like(imlist[0])
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
    if save_to is not None:
        cv2.imwrite(save_to, c)
    if need_return:
        return c

