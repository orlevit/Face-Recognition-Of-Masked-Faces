import os
import argparse
import multiprocessing
from script_config import ARCFACE_DATSETS_LOC


def run_multy(func, inputs):
    jobs = []
    #import pdb;  pdb.set_trace()
    for input in inputs:
        p = multiprocessing.Process(target=func, args=(input,))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

def aligned_output_dir(input):
    rest_path, dir_name = os.path.split(input)
    output_dir = os.path.join(rest_path, 'a' + dir_name)

    return output_dir

def train_input_dir(input):
    rest_path, dir_name = os.path.split(input)
    # rest_path, ds_name = os.path.split(rest_path)
    output_dir = os.path.join(ARCFACE_DATSETS_LOC, dir_name[1:])

    return output_dir

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory with input masked faces')
    # parser.add_argument('-o', '--output', type=str, help='Output directory.')
    # parser.add_argument('-e', '--image_extensions', default='.jpg,.bmp,.jpeg,.png',
    #                     type=str, help='The extensions of the images.')
    # parser.add_argument('-m', '--masks', default=ALL_MASKS, type=str, help='Which masks to create.')
    # parser.add_argument('-t', '--threshold', default=0.0, type=float,
    #                     help='The minimum confidence score for img2pose for face detection')

    return parser.parse_args()


