#!/usr/bin/env python3
# -*-encoding: utf-8-*-
# Author: Danil Kovalenko


import os
import h5py
import argparse
from typing import Tuple, Union

import numpy as np
from PIL import Image


def update_dataset(res: Union[None, np.ndarray],
                   f_path: str,
                   size: Tuple[int, int]) -> np.ndarray:
    with open(f_path, 'rb') as f:
        image = Image.open(f)
        image = image.resize(size)
        image_arr = np.array(image)
        image_arr.resize((1, *image_arr.shape))

        if res is None:
            return image_arr
        else:
            return np.append(res, image_arr)


def main(dir_path: str, size: Tuple[int, int], save_file_name: str) -> None:
    res = None
    for cur_dir, sub_dirs, files in os.walk(dir_path):
        for f_name in files:
            if f_name.endswith('.jpeg'):
                f_path = os.path.join(cur_dir, f_name)
                res = update_dataset(res, f_path, size)

    with h5py.File(save_file_name, 'w') as f:
        f.create_dataset('lines', data=res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path', type=str)
    parser.add_argument('--size', nargs=2, default=(128, 128))
    parser.add_argument('--save_file_name', type=str, default='dataset.h5')
    args = parser.parse_args()
    main(**args.__dict__)
