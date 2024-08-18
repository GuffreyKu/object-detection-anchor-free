# object-detection-anchor-free
## Flower Center Net
## Dataset
* coco format dataset.
* creat data folder, put data here.

## Install tool
* pytorch, torch_optimizer, opencv, imgaug

## How to use
* python trainer.py to train a model.
* python torch2onnx.py convert torch model to onnx model.
* python onnx2trt.py convert onnx model to tensorRT model.