# HATS-PyTorch

A *unofficial* PyTorch+CUDA implementation of "HATS: Histograms of Averaged Time 
Surfaces for Robust Event-based Object Classification", Sironi et al. 

Tested on Pytorch 1.6.0, CUDA 10.1

### Installation

- Clone this repository (add `--recursive` if you want to run the demo script)<br>
`git clone https://github.com/marcocannici/hats_pytorch.git`
- Install `hats_pytorch` (this also builds the CUDA kernels):
`cd hats_pytorch; python setup.py install`

You can test the implementation on the 
[N-Cars](https://www.prophesee.ai/2018/03/13/dataset-n-cars/) dataset by running 
the following command (you must have cloned the repository with `--recursive` 
for this to work):

`python demo/train.py --data_dir /path/to/ncars --batch_size 64`

Average time to extract representations from all the N-Cars training set on a 
GeForce GTX 1080ti is 1.24 ms/sample with batch_size 1, and 0.11 ms/sample 
with batch_size 64


### Usage

```python
from hats_pytorch import HATS
hats = HATS((100, 120), r=3, k=10, tau=1e9, delta_t=200e3, bins=1, fold=True)
hats.to('cuda:0')
histograms = hats(events, lengths)
```


### Disclaimer

This is unofficial code and, as such, the implementation may differ from the one
reported in the paper. If you find any error or difference with the paper, do not
hesitate to report it! :smiley:

