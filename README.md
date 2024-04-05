# Imagenet to miniimagenet labels
Provide three files
1. `hide_images.txt` list of images in the val+test set of miniimagenet, 
2. `hide_labels.txt` list of the 16+20 labels in the val+test set of miniimagenet,
3. `ilsvrc_cleaned.txt` the imagenet-1k train set image list stripped of ALL the samples that belong to the 16+20 val+test classes of miniimagenet.

## Motivations
In a meta-learning/few-shot learning setting there should be no overlap between train/val/test splits. So it is important that all the classes seen during training are not seen during the test phase.

In addition, I was able to find a [miniimagenet dataset version](https://www.kaggle.com/datasets/arjunashok33/miniimagenet) that has both the original resolution and the filename of imagenet-1k. [Ravi, Larochelle](https://openreview.net/forum?id=rJY0-Kcll) dataset [splits](https://github.com/mileyan/simple_shot/tree/master/split/mini) were consistent with the labels, but not with image names.

## Requirements
* `pandas`
* ilsvrc files (put in `src/ilsvrc`), in particular:
    * `LOC_val_solution.csv`
    * `train_cls.txt`

Both files can be found by downloading the imagenet-1k dataset from [kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description). I cannot provide them due to its licence.