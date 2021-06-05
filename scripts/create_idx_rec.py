import os
from shutil import copyfile
from script_helper import train_input_dir
from script_config import START_INSIGHT_ENV, IDX_REC_FUNC, CASIA_PROPERTY, PROPERTY, CASIA_LST_FILE, LST_FILE


def prerequisite_idx_rec(output_dirs):
    for output_dir in output_dirs:
            copyfile(CASIA_LST_FILE, os.path.join(output_dir, LST_FILE))
            copyfile(CASIA_PROPERTY, os.path.join(output_dir, PROPERTY))


def make_idx_rec(input):
    print('Start make idx&rec for: ', input)
    os.system(f'{START_INSIGHT_ENV} python {IDX_REC_FUNC} {input}')

    output_dir = train_input_dir(input)
    copyfile(os.path.join(input,'CASIA-WebFace.idx'), os.path.join(output_dir, 'train.idx'))
    copyfile(os.path.join(input,'CASIA-WebFace.rec'), os.path.join(output_dir, 'train.rec'))
    print('Finished make idx&rec for: ', input)


