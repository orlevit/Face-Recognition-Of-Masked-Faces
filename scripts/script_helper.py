import os
import time
import shutil
import argparse
import numpy as np
from glob import iglob
from script_config import ARCFACE_DATSETS_LOC, SBATCH, SLEEP_TIME, RESULTS_HEADERS, RESULTS_TARGET_FILES, \
    MODELS_DIRS_LIST, SLURM_LOGS_DIR, ARCFACE_DS_NAMES, NO_DATASET_TEST


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
    finished_jobs = []
    while len(finished_jobs) != line_number:
        finished_jobs = open(log_file).readlines()
        print(f'Processed jobs: {len(finished_jobs)}/{line_number}, content: {finished_jobs}')
        if 'FAIL\n' in finished_jobs:
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


def get_latests_results(prefix_input_files):
    results_file = [f for f in iglob(os.path.join(SLURM_LOGS_DIR, f'{prefix_input_files}-*.out'))]
    results_job_num = [int(os.path.split(f)[-1].split('-')[1]) for f in results_file]
    max_result = max(results_job_num)
    latest_jobs = [f for f in results_file if max_result == int(os.path.split(f)[-1].split('-')[1])]

    return latest_jobs


def fill_table(arr_results, latest_jobs):
    for f in latest_jobs:
        results = open(f).readlines()
        result = results[0].strip().split('/')[-1]
        first, middle, _, tested_db = result.strip().split("_")
        _, db_name = first.split("-")
        model_name = middle.split("-")[-1]
        model_name_idx = [i for i, one_model in enumerate(MODELS_DIRS_LIST) if model_name in one_model.split('/')[-1]]
        target_db_idx = [i for i, one_model in enumerate(ARCFACE_DS_NAMES) if tested_db in one_model]

        if len(model_name_idx) != 1 or len(target_db_idx) != 1:
            raise Exception(f'The length is not 1: {model_name_idx} or {target_db_idx}')

        # Not test this dataset
        #if tested_db == NO_DATASET_TEST:
        #   continue

        db_name_idx = RESULTS_TARGET_FILES.index(db_name)
        threshold = results[1].strip()
        accuracy = results[2].strip()
        auc = results[3].strip()

        # minues 1 because of the original dataset
        table_row = model_name_idx[0] * (len(MODELS_DIRS_LIST) -1) + target_db_idx[0]
        arr_results[table_row, 0] = model_name
        arr_results[table_row, 1] = tested_db
        arr_results[table_row, 2] = threshold
        db_loc_buff = len(RESULTS_TARGET_FILES) * db_name_idx
        arr_results[table_row, 3 + db_loc_buff] = accuracy
        arr_results[table_row, 4 + db_loc_buff] = auc


def organized_results(prefix_input_files, output_file):
    arr_results = np.empty([len(MODELS_DIRS_LIST) * len(ARCFACE_DS_NAMES), 7], dtype='O')
    latest_jobs = get_latests_results(prefix_input_files)
    fill_table(arr_results, latest_jobs)
    np.savetxt(output_file, arr_results, fmt='%s', delimiter=',', header=RESULTS_HEADERS, comments='') 


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory with input masked faces')

    return parser.parse_args()
