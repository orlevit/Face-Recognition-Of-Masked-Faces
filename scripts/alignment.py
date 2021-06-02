import os
import multiprocessing
from scripts.script_config import START_ENV, ALIGN_FUNC


def run_align(input_dir):
    print(f'start: {input_dir}')
    full_path, dir_name = os.path.split(os.path.normpath(input_dir[1]))
    output_dir = os.path.join(full_path, 'a'+dir_name)
    os.system(f'{START_ENV} python {ALIGN_FUNC} --input-dir {input_dir[1]} --output-dir {output_dir}')
    print(f'end: {input_dir}')


def align_mtcnn(masked_faces_dirs):
    jobs = []
    for input_dir in enumerate(masked_faces_dirs):
        # print(i)
        p = multiprocessing.Process(target=run_align, args=(input_dir,))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()