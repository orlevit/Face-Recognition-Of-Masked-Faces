import os
import multiprocessing
from shutil import copyfile
from scripts.script_helper import run_multy
from scripts.script_config import START_ENV, BIN_FUNC, ARCFACE_DATSETS_LOC


def prerequisite_bin(pairs_files):
    for input_dir, output_dirs in zip(pairs_files[0], pairs_files[1]):
        full_path, file_name = os.path.split(input_dir)
        for dst in output_dirs:
            copyfile(input_dir, os.path.join(dst, file_name))


def make_bin(input):
    rest_path, dir_name = os.path.split(input)
    rest_path, ds_name = os.path.split(rest_path)

    output_dir = os.path.join(ARCFACE_DATSETS_LOC, ds_name, dir_name)
    os.system(f'{START_ENV} python {BIN_FUNC} --data-dir {input} --output {output_dir} - -image-size 112, 112')