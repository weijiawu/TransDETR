# Rotated_ROIAlign
Rotated RoIAlign in pytorch, implemented with CUDA. This implementation is edited from [caffe operator](https://github.com/pytorch/pytorch/blob/master/caffe2/operators/roi_align_rotated_op.cc) and use pytorch's data interface.

# Usage
1. First complile the source file
```bash
    python setup.py build_ext --inplace
```
2. Then you can call the Rotated RoIAlign in python.
```python
    pooler_rotated=ROIAlignRotated((32,192), spatial_scale = (1.), sampling_ratio = 0)
    image_roi_bbox=pooler_rotated(<Input image>,<Input RoI>)
```

# Example
`test.py` is a example how to use the rotated RoIAlign.
The input image
![input image](IMG_0451.jpg)
the roi result
![output image](RotateAlign.png)
