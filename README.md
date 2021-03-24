# Implementation of F3-Net​ : Frequency in Face Forgery Network

## Note

This is a personal implementation of F3-Net , so there are lots of difference compared to the official version. To learn more details about F3-Net, please check the [paper](https://arxiv.org/abs/2007.09355) here.

Actually, I failed to reproduce the results of the paper for now. So all kinds of advices to improve model performance are welcomed and I'd like to discuss with you.



## Usage

#### Hyperparameter

Hyperparameters are in `train.py`.

| Variable name   | Description                             |
| --------------- | --------------------------------------- |
| dataset_path    | The path of dataset, support FF++ only. |
| pretrained_path | The path of pretrained Xception model.  |
| batch_size      | 128 in paper.                           |
| max_epoch       | how many epochs to train the model.     |
| loss_freq       | print loss after how many iterations    |
| mode            | mode of the network, see details below. |



#### Load a pretrained Xception

Download *Xception* model trained on ImageNet (through this [link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) or use your own pretrained *Xception*.

Then modify the `pretrained_path`  variable.



#### Use FF++ dataset

The dataset related function is designed for `FaceForensics++`  dataset. Check this [github repo](https://github.com/ondyari/FaceForensics) or [paper](https://arxiv.org/abs/1901.08971) for more details of the dataset.

After preprocessing, the data should be organized as following:

```
|-- dataset
|   |-- train
|   |   |-- real
|   |   |	|-- 000
|   |   |	|	|-- frame0.jpg
|   |   |	|	|-- frame1.jpg
|   |   |	|	|-- ...
|   |   |	|-- 001
|   |   |	|-- ...
|   |   |-- fake
|   |   	|-- Deepfakes
|   |   	|	|-- 000_167
|   |		|	|	|-- frame0.jpg
|   |		|	|	|-- frame1.jpg
|   |		|	|	|-- ...
|   |		|	|-- 001_892
|   |		|	|-- ...
|   |   	|-- Face2Face
|   |		|	|-- ...
|   |   	|-- FaceSwap
|   |   	|-- NeuralTextures
|   |-- valid
|   |	|-- real
|   |	|	|-- ...
|   |	|-- fake
|   |		|-- ...
|   |-- test
|   |	|-- ...
```



#### Model mode

There are four modes supported in F3-Net​.

| Mode(string) |                                                         |
| ------------ | ------------------------------------------------------- |
| 'FAD'        | Use FAD branch only.                                    |
| 'LFS'        | Use LFS branch only.                                    |
| 'Both'       | Use both of branches and concate before classification. |
| 'Mix'        | Use both of branches and MixBlock.                      |

 

#### Run

Environment:

**Pytorch, torchvision, numpy, sklearn, pillow** are needed.

**To train the model**

`python train.py`



## Architecture Details

I've written a doc about implementation details and my confusion during coding. Check `details.md` if you're interested. [Still working on it]



## Reference

Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. arXiv preprint arXiv:2007.09355, 2020

[Paper Link](https://arxiv.org/abs/2007.09355)