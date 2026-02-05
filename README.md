# [ICLR 2026] Reducing class-wise performance disparity via margin regularization
_________________

Official implementation of the ICLR 2026 paper

**Abstract**: Deep neural networks often exhibit substantial disparities in class-wise accuracy, even when trained on class-balanced data, posing concerns for reliable deployment. While prior efforts have explored empirical remedies, a theoretical understanding of such performance disparities in classification remains limited. In this work, we present Margin Regularization for Performance Disparity Reduction (MR^2), a theoretically principled regularization for classification by dynamically adjusting margins in both the logit and representation spaces. Our analysis establishes a margin-based, class-sensitive generalization bound that reveals how per-class feature variability contributes to error, motivating the use of larger margins for hard classes. Guided by this insight, MR^2 optimizes per-class logit margins proportional to feature spread and penalizes excessive representation margins to enhance intra-class compactness. Experiments on seven datasets, including ImageNet, and diverse pre-trained backbones (MAE, MoCov2, CLIP) demonstrate that MR^2 not only improves overall accuracy but also significantly boosts hard class performance without trading off easy classes, thus reducing performance disparity.


## CIFAR-100

### Dependency

The code is built with following libraries:

- [PyTorch1.2](https://pytorch.org/) 
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [scikit-learn](https://scikit-learn.org/stable/)

### Dataset

- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html). 

### Training 

We provide several training examples with this repo:

- To train with our margin regularization framework

```bash
bash mrf_train.sh
```



## Imagenet and Other Datasets

### Dependency

```bash
conda env create
conda activate mrf
```

### Add directory to PYTHONPATH:

```bash
cd MRF-Imagenet
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Download data

When necessary, please refer to [DATASETS.md](MRF-Imagenet/DATASETS.md) for instructions on how to download datasets.

### Training 

We provide several training examples with this repo:

- To train with our margin regularization framework

```bash
bash mrf_train.sh
```


 ## Citation
 
If you find this repository useful, please consider citing the following paper:

```
@inproceedings{zhu2026reducing,
  title = {Reducing class-wise performance disparity via margin regularization},
  author = {Zhu, Beier and Zhao, Kesen and Cui, Jiequan and Sun, Qianru and Zhou, Yuan and Yang, Xun and Zhang, Hanwang},
  booktitle = {International Conference on Learning Representations},
  year = {2026},
}
```