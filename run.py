import os
import sys

from src.parsers import Ilsvrc, Mini
from src.consts import Const as _C

if __name__=="__main__":
    # check file are in the directory
    root = os.path.dirname(os.path.realpath(__file__))
    ilsvrc_files = os.listdir(os.path.join(root, "res", "ilsvrc"))
    if _C.ILSVRC_TRAIN_FILE in ilsvrc_files and _C.ILSVRC_VAL_FILE in ilsvrc_files:
        pass
    else:
        print(f"ERROR: {_C.ILSVRC_TRAIN_FILE} and {_C.ILSVRC_VAL_FILE} must be in $ROOT/res/ilsvrc. Follow README")
        sys.exit(-1)

    # init parsers
    p_ilsvrc = Ilsvrc(root)
    p_mini = Mini(root)

    # parse everything
    ilsvrc_train = p_ilsvrc.get_train_imgs()
    ilsvrc_val = p_ilsvrc.get_val_imgs()
    mini_all = p_mini.get_all_imgs()
    mini_train = p_mini.get_train_imgs()
    mini_val = p_mini.get_val_imgs()
    mini_test = p_mini.get_test_imgs()

    # this to prove that all the miniimagenet images come from ILSVRC train split
    nimg = _C.MINI_N_IMG_PER_CLASS
    
    ilsvrctrain_and_minitrain = set(ilsvrc_train) & set(mini_train)
    print(f"{len(ilsvrctrain_and_minitrain)}/{(nimg * _C.MINI_N_CLASS_TRAIN)}, mini train in ilsvrc train")

    ilsvrctrain_and_minival = set(ilsvrc_train) & set(mini_val)
    print(f"{len(ilsvrctrain_and_minival)}/{(nimg * _C.MINI_N_CLASS_VAL)}, mini val in ilsvrc train")

    ilsvrctrain_and_minitest = set(ilsvrc_train) & set(mini_test)
    print(f"{len(ilsvrctrain_and_minitest)}/{(nimg * _C.MINI_N_CLASS_TEST)}, mini test in ilsvrc train")

    # save a file of val/test samples and a clean version of ILSVRC train
    outfolder = os.path.join(root, "output")
    if not os.path.exists(outfolder): os.makedirs(outfolder)

    print("Filtering imagenet-1k. May take some time...")
    mini_val_test_imgs = set(mini_val) | set(mini_test)
    mini_val_test_lbls = set([i.rsplit("_", -1)[0] for i in mini_val_test_imgs])
    ilsvrc_cleaned = list(filter(lambda x: not any(x.startswith(l) for l in mini_val_test_lbls), set(ilsvrc_train)))

    with open(os.path.join(outfolder, "hide_images.txt"), "w") as f:
        for name in mini_val_test_imgs:
            f.write(f"{name}\n")

    with open(os.path.join(outfolder, "hide_labels.txt"), "w") as f:
        for name in mini_val_test_lbls:
            f.write(f"{name}\n")

    with open(os.path.join(outfolder, "ilsvrc_cleaned.txt"), "w") as f:
        for name in ilsvrc_cleaned:
            f.write(f"{name}\n")