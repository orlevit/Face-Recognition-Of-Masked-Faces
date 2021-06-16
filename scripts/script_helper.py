import os
import time
import argparse
import multiprocessing
from script_config import ARCFACE_DATSETS_LOC, SBATCH, SLEEP_TIME


def run_multy(func, inputs):
    jobs = []
    #import pdb;  pdb.set_trace()
    for input in inputs:
        p = multiprocessing.Process(target=func, args=(input,))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

def aligned_output_dir(inputs):
    output_dirs = [] 
    for input in inputs:
	    rest_path, dir_name = os.path.split(input)
	    output_dirs.append(os.path.join(rest_path, 'a' + dir_name))

    return output_dirs

def train_input_dir(inputs):
    output_dirs = [] 
    for input in inputs:
	    rest_path, dir_name = os.path.split(input)
	    # rest_path, ds_name = os.path.split(rest_path)
	    output_dirs.append(os.path.join(ARCFACE_DATSETS_LOC, dir_name[1:]))

    return output_dirs

def wait_until_jobs_finished(log_file, line_number):
    print(log_file)
    while len(open(log_file).readlines()) != line_number:
       print('line_number',line_number,'content ', open(log_file).readlines())
       time.sleep(SLEEP_TIME)
    
    if 'FAIL\n' in open(log_file).readlines():
       raise ValueError(f'{log_file} - Job failed!') 

def delete_create_file(log_file):
    print('Existsa fiel1',os.path.exists(log_file))
    if os.path.exists(log_file):
       os.remove(log_file)

    open(log_file, 'a').close()
    print('Existsa fiel2',os.path.exists(log_file))

def sbatch(sbatch_file, mem, job_name, jobs_number, input_str):
    os.system(f'{SBATCH.format(mem, job_name, jobs_number)} {sbatch_file} {input_str}')

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


