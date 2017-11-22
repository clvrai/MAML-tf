# MAML-tf
Implementation of MAML in Tensorflow

## Introduction

## Prerequisites
- Python==2.7
- Tensorflow==1.4.0

## Usage 

### Regression

Train 5-shot regreesion model:
```bash
python main.py --dataset sin --K 5 --num_updates 1 --norm None --is_train
```

Details about the training FLAGs
```
--K: draw K samples as meta-train and meta-val
--model_type: for regression, I only use fully connected layers
--loss_type: for regression, I use MeanSquareError as loss criterion
--num_updates: do `num_updates` graident step for meta-step
--norm: use batch_norm or not
--alpha: learning rate for meta-train (same notation as the paper)
--beta: learning rate for meta-val (same notation as the paper)
```

Evalaute the model (either specify the directory of the checkpoint or the checkpoint itself):
```bash
python main.py --dataset sin --K 5 --num_updates 5 --norm None --restore_checkpoint PATH_TO_CHECKPOINT
```

Details about the evaluation FLAGs (some are overlapped with training)
```
--K: draw K samples as meta-train and meta-val
--num_updates: do `num_updates` graident step for meta-step
--restore_checkpoint: specify the path to the checkpoint
--restore_dir: specify the path to the directory of the checkpoint (directly choose the latest one)
--test_sample: number of testing samples
--draw: visualize or not
```

## Results

## What's inside the training?

### Tricks

## Related Work and Reference
- Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks
