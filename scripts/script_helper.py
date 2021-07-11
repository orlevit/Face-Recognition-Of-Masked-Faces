import os
import time
import shutil
import argparse
import sys
from scripts.script_config import ARCFACE_DATSETS_LOC, SBATCH, SLEEP_TIME


# def run_multy(func, inputs):
#    jobs = []
#    for input in inputs:
#        p = multiprocessing.Process(target=func, args=(input,))
#        jobs.append(p)
#        p.start()
#
#    for job in jobs:
#        job.join()

def aligned_output_dir(inputs):
    output_dirs = []
    for one_input in inputs:
        rest_path, dir_name = os.path.split(one_input)
        output_dirs.append(os.path.join(rest_path, 'a' + dir_name))

    return output_dirs


def bin_output_dir(inputs):
    output_dirs = []
    for one_input in inputs:
        rest_path, dir_name = os.path.split(one_input)
        _, base_dir_name = os.path.split(rest_path)
        output_dir = os.path.join(ARCFACE_DATSETS_LOC, dir_name[1:])
        delete_create_dir(output_dir)
        output_dirs.append(os.path.join(output_dir, base_dir_name) + ".bin")

    return output_dirs


def idx_rec_output_dir(inputs):
    output_dirs = []
    for one_input in inputs:
        rest_path, dir_name = os.path.split(one_input)
        _, base_dir_name = os.path.split(rest_path)
        output_dir = os.path.join(ARCFACE_DATSETS_LOC, dir_name[1:])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_dirs.append(output_dir)

    return output_dirs


def wait_until_jobs_finished(log_file, line_number):
    print(log_file)
    finished_jobs_number = len(open(log_file).readlines())
    while finished_jobs_number != line_number:
        print(f'line_number: {finished_jobs_number}/{line_number}, {content: {open(log_file).readlines())}')
        if 'FAIL\n' in open(log_file).readlines():
            raise ValueError(f'{log_file} - Job failed!')
        time.sleep(SLEEP_TIME)


def delete_create_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def delete_create_file(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    open(log_file, 'w').close()


def sbatch(sbatch_file, mem, job_name, jobs_number, input_str):
    os.system(f'{SBATCH.format(mem, job_name, jobs_number)} {sbatch_file} {input_str}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory with input masked faces')

    return parser.parse_args()
