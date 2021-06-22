import os
from glob import iglob
from itertools import groupby
from scripts.script_helper import sbatch, delete_create_file, wait_until_jobs_finished
from scripts.script_config import TEST_MEM, TEST_JOBS_NAME, TEST_SBATCH_FILE, ARCFACE_ENV, TEST_RESULTS_FILE, \
    TEST_TRACK_FILE, MODELS_LIST, ARCFACE_DATSETS_LOC, ARCFACE_DS_NAMES, ARCDACE_VALIDATON_DS


def get_bin_test_files():
    test_files = []
    datasets_dirs = [os.path.join(ARCFACE_DATSETS_LOC, path) for path in os.listdir(ARCFACE_DATSETS_LOC)]
    for ds_dir in datasets_dirs:
        for test_path in iglob(os.path.join(ds_dir, '*.bin')):
            rest_path, file_name = os.path.split(test_path)
            _, dir_name = os.path.split(rest_path)
            if dir_name in ARCFACE_DS_NAMES and file_name not in ARCDACE_VALIDATON_DS:
                test_files.append(test_path)

    grouped_input = [list(g) for k, g in groupby(test_files, lambda s: s.split('/')[-2])]
    return grouped_input


def make_test():
    grouped_input = get_bin_test_files()
    delete_create_file(TEST_TRACK_FILE)
    items_number = sum([len(list_input) for list_input in grouped_input])
    input_number = items_number * len(MODELS_LIST)

    input_str = ''
    for model, threshold in zip(MODELS_LIST, THRESHOLDS):
        for input_files_list in grouped_input:
            for input_file in input_files_list:
                model_name = model.split('/')[-2]
                data_dir, target = os.path.split(input_file)
                _, dir_name = os.path.split(data_dir)
                roc_name = f'{model_name}_{dir_name}'
                input_str += f'{ARCFACE_ENV} {data_dir} {target} {model} ' \
                             f'{roc_name} {threshold} {TEST_RESULTS_FILE} {TEST_TRACK_FILE} '

    sbatch(TEST_SBATCH_FILE, TEST_MEM, TEST_JOBS_NAME, input_number, input_str)

    wait_until_jobs_finished(TEST_TRACK_FILE, input_number)


if __name__ == '__main__':
    make_test()
