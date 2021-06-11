import os
from shutil import copyfile
from script_helper import train_input_dir
from script_config import START_INSIGHT_ENV, BIN_FUNC


def prerequisite_bin(pairs_files):
    for input_dir, output_dirs in zip(pairs_files[0], pairs_files[1]):
        full_path, file_name = os.path.split(input_dir)
        for dst in output_dirs:
            copyfile(input_dir, os.path.join(dst, file_name))


def make_bin(input):
    print('Start make bin for: ', input)

    output_dir = train_input_dir(input)

    os.system(f'{START_INSIGHT_ENV} python {BIN_FUNC} --data-dir {input} --output {output_dir} - -image-size 112, 112')
    print('Finished make bin for: ', input)
