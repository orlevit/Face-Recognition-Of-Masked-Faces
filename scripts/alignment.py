import os
import multiprocessing
from script_helper import aligned_output_dir
from script_config import START_MTCNN_ENV, ALIGN_FUNC


def align_mtcnn(input_dir):
    #import pdb;  pdb.set_trace()
    print(f'Start align: {input_dir}')
    output_dir = aligned_output_dir(input_dir)
    os.system(f'{START_MTCNN_ENV} python {ALIGN_FUNC} --input-dir {input_dir} --output-dir {output_dir}')
    print(f'End align: {input_dir}')


# def align_mtcnn(masked_faces_dirs):
#     run_multy(run_align, masked_faces_dirs)
#     jobs = []
#     for input_dir in enumerate(masked_faces_dirs):
#         # print(i)
#         p = multiprocessing.Process(target=run_align, args=(input_dir,))
#         jobs.append(p)
#         p.start()
#
#     for job in jobs:
#         job.join()
