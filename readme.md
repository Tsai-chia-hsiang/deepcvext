# handmade some OpenCV-python extension for machine/deep learning 
The handmade extension corporate with OpenCV and some Python native libraries for machine/deep learning on computer vision purposes 
 
## Motivation

Manipulate/Visualization images using pure OpenCV, not alum/PIL including. 
- Ensure non-deeplearning framework part of image processes in OpenCV platform.
  - But maybe some floating point error or numpy behavior still different from C++. So... can view this as just a fun side project if you really want everything align to C++ operation.

## [convert.py](./convert.py): 
Conversion between OpenCV images (NumPy arrays) and images from other frameworks with custmize scaling function:
### Convert image (np.ndarray) to framework 
  
  Default normalize setting: $/255$
  
  If you want to convert a batch of images, please compose the image as a list of np.ndarray.
  
  - PyTorch torch.Tensor: ```img2tensor()```
    - $H\times W \times C \to (B=1)\times C\times H \times W$
    
  - jax jax.numpy.ndarray : ```img2jnp()```
    - jax hold same axis order for single image as opencv $H\times W\times C$  
    - jnp.array(img)
    - Note that you need to specific the device since jax.array() will automatically map the array into device
      - Set from `device` argument: ```tuple[str, int]```
      - default is ```('gpu', 0)```
  
### Convert framework to image (np.ndarray) 
If given a batch images ($N_{dim} = 4$), will return list of np.ndarray, each element of returned list is the corresponding image.
- Please viewing ```_img_debatch()``` for more detail

The defualt scale-back function is 
```python=
_TO_IMG_ = lambda x: np.clip(x, 0, 1)*255
```
  - PyTorch torch.Tensor: ```tensor2img()```
  - jax: jax.numpy.ndarray: ```jnp2img()```



## [box.py](./box.py): Bounding box (bbox) tools, support bboxes in:
  - list of list
  - np.ndarray
  - torch.Tensor

format

## [utils.py](./utils.py): the cv2 tool
I have to confess this is redundant as library. I write this lib just for my master thesis. So...

## [draw.py](./draw.py): visualization tool
Pure OpenCV visualization method
- ```canvas()```:
  - To demostrate image in grid for the give row 
  - Eequire all image at same size
- ```draw_box()```:
  - bboxes visualization
    - But I find Ultralytics and torchvision also provide same function as well after I design this, so kind of redundant as well.
- ```draw_segmentation_map()```:
  - coloring the segmenation map
  - If you don't passing `cid_color_map (dict[int, tuple[int,int,int]])`, it will use random colors generated by:
    - ```
      _ = np.random.default_rng(42)
      _Ccolors = np.random.randint(0, 256, size=(500, 3), dtype=np.uint8)
      ```
