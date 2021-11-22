import os
import sys
import shutil
import inspect
from glob import iglob
from itertools import groupby

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(currentdir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(-1, parent_dir)
sys.path.insert(-1, parent_parent_dir)

from scripts.script_helper import sbatch, delete_create_file, wait_until_jobs_finished, organized_results
from scripts.script_config import TEST_DIFF_MEM, TEST_DIFF_JOBS_NAME, TEST_DIFF_SBATCH_FILE, ARCFACE_ENV, \
    TEST_DIFF_COMMANDS_FILE, TEST_DIFF_TRACK_FILE, MODELS_DIRS_LIST, ARCFACE_DATSETS_LOC, ARCFACE_DS_NAMES, \
    ARCFACE_VALIDATON_DS, NOMASK_DATA_LOC, TEST_DIFF_RESULTS_FILE, TEST_DIFF_ROC


def get_bin_test_files():
    test_files = []
    datasets_dirs = [os.path.join(ARCFACE_DATSETS_LOC, path) for path in os.listdir(ARCFACE_DATSETS_LOC)]
    for ds_dir in datasets_dirs:
        for test_path in iglob(os.path.join(ds_dir, '*.bin')):
            rest_path, file_name = os.path.split(test_path)
            _, dir_name = os.path.split(rest_path)
            if dir_name in ARCFACE_DS_NAMES and file_name not in ARCFACE_VALIDATON_DS:
                test_files.append(test_path)

    grouped_input = [list(g) for k, g in groupby(test_files, lambda s: s.split('/')[-2])]
    return grouped_input


def read_model_file(model_dir, file_ending):
    files_in_dir = []
    for f in iglob(os.path.join(model_dir, file_ending)):
        files_in_dir.append(f)

    if len(files_in_dir) != 1:
        raise Exception(f"There isn't only 1 file in directory: {model_dir} ending with: {file_ending} "
                        f"and the files are: {files_in_dir}")
    return files_in_dir[0]


def get_model(model_dir):
    model_full_name = read_model_file(model_dir, "*.params")
    model_full_threshold = read_model_file(model_dir, "*threshold.txt")

    model_name_raw = model_full_name.split('/')[-1].split('.')[0].split('-')
    model_name = model_name_raw[0] + ',' + str(int(model_name_raw[1]))
    with open(model_full_threshold, "r") as txt_file:
        threshold_f_content = txt_file.read()

    # NOTICE: working only with *ONE* threshold in the file
    threshold = float(threshold_f_content.split('_')[1])
    model_loc = os.path.join(model_dir, model_name)

    return model_loc, threshold


def make_test_diff():
    if os.path.exists(TEST_DIFF_ROC):
       shutil.rmtree(TEST_DIFF_ROC)
    os.makedirs(TEST_DIFF_ROC)

    grouped_input = get_bin_test_files()
    delete_create_file(TEST_DIFF_TRACK_FILE)
    items_number = sum([len(list_input) for list_input in grouped_input])
    input_number = items_number * len(MODELS_DIRS_LIST)

    input_str = ''
    # TODO: create manually the .bin files from the new pairs.txt files for nomasks
    # TODO: test manually the threshold on nomask casia and add it to a threshold file!!!!!!!!!!!!!!!!!!
    # TODO: move the relevant files to the scripts relevant location
    for model_dir in MODELS_DIRS_LIST:
        model, threshold = get_model(model_dir)
        for input_files_list in grouped_input:
            for input_file in input_files_list:
                model_name = model.split('/')[-2]
                data_dir_mask, target = os.path.split(input_file)
                _, dir_name = os.path.split(data_dir_mask)
                target_name_only = target.split('.')[0]
                roc_name = os.path.join(TEST_DIFF_ROC, f'diff-{target_name_only}_{model_name}-nomask_{dir_name}')
                input_str += f'{ARCFACE_ENV} {data_dir_mask} {NOMASK_DATA_LOC} {target_name_only} {model} ' \
                             f'{roc_name} {threshold} {TEST_DIFF_COMMANDS_FILE} {str(TEST_DIFF_TRACK_FILE)} '

    sbatch(TEST_DIFF_SBATCH_FILE, TEST_DIFF_MEM, TEST_DIFF_JOBS_NAME, input_number, input_str)

    wait_until_jobs_finished(TEST_DIFF_TRACK_FILE, input_number)

    organized_results(os.path.basename(__file__).rsplit('.')[0], TEST_DIFF_RESULTS_FILE)


if __name__ == '__main__':
    make_test_diff()
