import os
from shutil import copyfile
from script_helper import delete_create_file, wait_until_jobs_finished, sbatch, idx_rec_output_dir
from script_config import ARCFACE_ENV, CASIA_PROPERTY, CASIA_LST_FILE, LST_FILE, IDX_REC_MEM, IDX_REC_JOBS_NAME, IDX_REC_FILE, IDX_REC_SBATCH_FILE


def prerequisite_idx_rec(output_dirs):
    for output_dir in output_dirs:
            print(output_dir)
            copyfile(CASIA_LST_FILE, os.path.join(output_dir, LST_FILE))
            copyfile(CASIA_PROPERTY, os.path.join(output_dir, PROPERTY))


def make_idx_rec(inputs):
    delete_create_file(IDX_REC_FILE)

    env = [ARCFACE_ENV] * len(inputs)
    output_dir = idx_rec_output_dir(inputs)
    file = [IDX_REC_FILE] * len(inputs)
    print(output_dir)
    print(inputs)
    input_str = ''
    for i, j, k, l in zip(env, inputs, output_dir, file):
       input_str += f'{i} {j} {k} {l} '

    sbatch(IDX_REC_SBATCH_FILE, IDX_REC_MEM, IDX_REC_JOBS_NAME, len(inputs), input_str)
    
    wait_until_jobs_finished(IDX_REC_FILE, len(inputs))

#def make_idx_rec(input):
#    print('Start make idx&rec for: ', input)
#    os.system(f'{START_INSIGHT_ENV} python {IDX_REC_FUNC} {input}')
#
#    output_dir = train_input_dir(input)
#    copyfile(os.path.join(input,'CASIA-WebFace.idx'), os.path.join(output_dir, 'train.idx'))
#    copyfile(os.path.join(input,'CASIA-WebFace.rec'), os.path.join(output_dir, 'train.rec'))
#    print('Finished make idx&rec for: ', input)
