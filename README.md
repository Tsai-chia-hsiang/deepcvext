# PyTorch & OpenCV-python ext
The handmade PyTorch extension corporate with OpenCV and some Python native libraries for deep learning on computer vision purposes 

- convert.py:
  - ```tensor2img``` : 
    - Convert image(s) in PyTorch Tensor format $(B \times)  C\times H\times W$ to numpy array or list of numpy array $H\times W\times C \text{ or } \underbrace{[(H\times W\times C), ...]}_{B}$
  
  - ```img2tensor``` : 
    - Convert a single image in OpenCV numpy ndarray $H\times W\times C$ to PyTorch Tensor $(1 \times)  C\times H\times W$, normalize the value by dviding by 255

- box.py: Bounding box (bbox) tools, support bboxes in:
  - list of list
  - np.ndarray
  - torch.Tensor
  
  formats
