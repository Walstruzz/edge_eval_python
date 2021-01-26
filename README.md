## Edge Eval Python
A python implementation of [edge eval](https://github.com/s9xie/hed_release-deprecated/tree/master/examples/eval).

## Requirements
* Python3
* Numpy
* Scipy >= 1.6.0
* g++
* Matplotlib
* 

## Install
### 1. clone repository
```shell
git clone https://github.com/Walstruzz/edge_eval_python
cd edge_eval_python
```

### 2. compile cxx library
Most of the code in this folder is copied from [davidstutz/extended-berkeley-segmentation-benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark/tree/master/source).
```shell
cd cxx/src
source build.sh
```

## Usage
### 1. save your results
```shell
from scipy.io import savemat

key = "result"
result = your_method(image)
savemat(save_name, {key: image})
```

### 2.eval
```shell
python main.py --alg "HED" --model_name_list "hed" --result_dir examples/hed_result \
--save_dir examples/hed_eval_result --gt_dir examples/bsds500_gt --key result \
--file_format .mat --workers -1
```

