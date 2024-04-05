import os
import pandas as pd

from abc import abstractmethod
from typing import List

from src.consts import Const as _C


class Parser:
    def __init__(self, root: str):
        if not os.path.exists(root):
            raise ValueError(f"ERROR: root is not correct")
        
        self.root = root

    @abstractmethod
    def get_train_imgs(self, rpath: List[str]) -> List[str]:
        ...

    @abstractmethod
    def get_val_imgs(self, rpath: List[str]) -> List[str]:
        ...


class Ilsvrc(Parser):

    def __init__(self, root: str):
        super().__init__(root)

    def get_train_imgs(self, rpath: List[str]=["res", "ilsvrc", _C.ILSVRC_TRAIN_FILE]) -> List[str]:
        with open(os.path.join(self.root, *rpath), "r") as f:
            lines = [line.strip() for line in f]

        if not len(lines) == _C.ILSCRC_N_TRAIN:
            raise ValueError(f"The number of training files should be {_C.ILSCRC_N_TRAIN}. Yours is {len(lines)}")

        filenames = [f"{(l.split('/')[1]).split()[0]}.JPEG" for l in lines]
        return filenames
    
    def get_val_imgs(self, rpath: List[str]=["res", "ilsvrc", _C.ILSVRC_VAL_FILE]) -> List[str]:
        df = pd.read_csv(os.path.join(self.root, *rpath))
        labels = list(df["PredictionString"].str.split().str.get(0).values)
        img_id = list(df["ImageId"].str.split("_val").str.get(1).values)

        filenames = []
        for label, image in zip(labels, img_id):
            filenames.append(f"{label}{image}.JPEG")
        
        return filenames


class Mini(Parser):

    def __init__(self, root: str):
        super().__init__(root)

    def get_all_imgs(self, rpath: List[str]=["res", "mini", _C.MINI_SUBSET_FILE]) -> List[str]:
        df = pd.read_csv(os.path.join(self.root, *rpath), index_col=0)
        filenames = list(df["filename"].values)

        correct_n = _C.MINI_N_IMG_PER_CLASS * 100
        if not len(set(filenames)) == correct_n:
            raise ValueError(f"Number of unique images found {len(set(filenames))} != {correct_n}")
        
        return filenames
    
    def __get_split_imgs(self, rpath: List[str]) -> List[str]:
        # read all the miniimagenet across all the splits
        all_imgs = self.get_all_imgs()

        # get the list of labels from the current selected set
        df_split = pd.read_csv(os.path.join(self.root, *rpath))
        split_labels = set(list(df_split["label"].values))

        # filter for those that starts with one of the 64/16/20 labels
        filenames = list(filter(lambda x: any(x.startswith(c) for c in split_labels), all_imgs))

        # check correct number of elements
        n_correct = _C.MINI_N_IMG_PER_CLASS
        if "train" in rpath[-1]:
            n_correct *= _C.MINI_N_CLASS_TRAIN
        elif "val" in rpath[-1]:
            n_correct *= _C.MINI_N_CLASS_VAL
        else:
            n_correct *= _C.MINI_N_CLASS_TEST

        if not len(set(filenames)) == n_correct:
            raise ValueError(f"Number of unique images found {len(set(filenames))} != {n_correct}")

        return filenames

    def get_train_imgs(self, rpath: List[str]=["res", "mini", _C.MINI_TRAIN_FILE]) -> List[str]:
        filenames = self.__get_split_imgs(rpath)
        return filenames

    def get_val_imgs(self, rpath: List[str]=["res", "mini", _C.MINI_VAL_FILE]) -> List[str]:
        filenames = self.__get_split_imgs(rpath)
        return filenames
    
    def get_test_imgs(self, rpath: List[str]=["res", "mini", _C.MINI_TEST_FILE]) -> List[str]:
        filenames = self.__get_split_imgs(rpath)
        return filenames