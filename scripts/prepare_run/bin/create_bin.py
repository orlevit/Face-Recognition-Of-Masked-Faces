import os
from shutil import copyfile
from script_helper import bin_output_dir, wait_until_jobs_finished, delete_create_file, sbatch
from script_config import ARCFACE_ENV, BIN_MEM, BIN_FILE, SLEEP_TIME, BIN_JOBS_NAME, BIN_SBATCH_FILE


def prerequisite_bin(pairs_files):
    for input_dir, output_dirs in zip(pairs_files[0], pairs_files[1]):
        full_path, file_name = os.path.split(input_dir)
        for dst in output_dirs:
            copyfile(input_dir, os.path.join(dst, 'pairs.txt'))


def make_bin(inputs):
    delete_create_file(BIN_FILE)

    env = [ARCFACE_ENV] * len(inputs)
    output_dir = bin_output_dir(inputs)
    file = [BIN_FILE] * len(inputs)

    input_str = ''
    for i, j, k, l in zip(env, inputs, output_dir, file):
        input_str += f'{i} {j} {k} {l} '

    sbatch(BIN_SBATCH_FILE, BIN_MEM, BIN_JOBS_NAME, len(inputs), input_str)

    wait_until_jobs_finished(BIN_FILE, len(inputs))
