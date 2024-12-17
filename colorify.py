from shared.utils import load_img_array
from tqdm import tqdm

import os 
import json 
from typing import Literal 

import numpy as np 
import cv2 as cv 
from scipy.spatial import KDTree

import re

import multiprocessing

import argparse

from shared.color_schemes import schemes
from shared.utils import load_img_array

from PIL import Image 

from pathlib import Path

def kdtree_LUT(image, palette) -> np.ndarray:
    """
    couldn't get actual LUTs working so here's my hack. this will iterate over each pixel's RGB value, and do a nearest neighbor search using a palette k-d tree.
    """
    palette_tree = KDTree(palette)
    original_shape = image.shape
    image = image.reshape(-1, 3)

    new = np.ones(image.shape)

    for i in range(image.shape[0]):
        pixel = image[i]
        _, idx = palette_tree.query(pixel)
        new[i] = palette[idx]

    return new.reshape(original_shape).astype(np.uint8)

def kdtree_LUT_smooth(image, palette) -> np.ndarray:
    """
    couldn't get actual LUTs working so here's my hack. this will iterate over each pixel's RGB value, and do a nearest neighbor search using a palette k-d tree.
    """
    mean_col = lambda c1, c2: (np.mean([c1[0], c2[0]]), np.mean([c1[1], c2[1]]), np.mean([c1[2], c2[2]]))

    palette_tree = KDTree(palette)
    original_shape = image.shape
    image = image.reshape(-1, 3)

    new = np.ones(image.shape)

    for i in range(image.shape[0]):
        pixel = image[i]
        _, idx = palette_tree.query(pixel)
        new[i] = mean_col(palette[idx], pixel)

    return new.reshape(original_shape).astype(np.uint8)

def _apply_lut(args): # wrapper function for multithreaded processing
    return kdtree_LUT(*args)
    
def kdtree_LUT_parallel(image, palette) -> np.ndarray:
    """divides image based on chunks to speed things up somewhat"""
    # divide image into chunks
    num_proc = multiprocessing.cpu_count() // 2
    chunk_size = image.shape[0] // num_proc 
    chunks = [
        image[i:i+chunk_size] for i in range(0, image.shape[0], chunk_size)
    ]

    # add leftover
    if len(chunks) > num_proc:
        chunks[-2] = np.vstack([chunks[-2], chunks[-1]])
        chunks = chunks[:-1]


    with multiprocessing.Pool(num_proc) as pool:
        proc_chunks = list(
            tqdm(
                pool.imap(
                    _apply_lut,
                    [(chunk, palette) for chunk in chunks] # scuffed hacky fix but apparently can't pickle lambdas for multiprocessing
                ),
                total=len(chunks)
            )
        )

    return np.vstack(proc_chunks).astype(np.uint8)

colorscheme_choices = [k for k in schemes.keys()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 

    parser.add_argument("image_path", help="path to image to convert")
    parser.add_argument("colorscheme", default="gruvbox", help="color scheme, choices are {}".format(",".join(colorscheme_choices)))
    parser.add_argument("--blur", action='store_true', help="quick gaussian blur before color application")

    args = parser.parse_args()
    image = load_img_array(args.image_path)[:,:,:3]
    if args.blur:
        image = cv.GaussianBlur(image, (3, 3), 0)
    palette = schemes[args.colorscheme]

    out_image = kdtree_LUT_parallel(image, palette)

    Image.fromarray(out_image).save(args.colorscheme + "_" + Path(args.image_path).stem + ".jpg")

    