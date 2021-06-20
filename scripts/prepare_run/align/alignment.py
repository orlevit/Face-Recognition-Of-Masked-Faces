# import multiprocessing
from script_helper import aligned_output_dir, delete_create_file, sbatch, wait_until_jobs_finished
from script_config import MTCNN_ENV, ALIGN_MEM, ALIGN_JOBS_NAME, ALIGN_FILE, ALIGN_SBATCH_FILE


def make_align(inputs):
    print(inputs, len(inputs))
    delete_create_file(ALIGN_FILE)
    env = [MTCNN_ENV] * len(inputs)
    output_dir = aligned_output_dir(inputs)
    file = [ALIGN_FILE] * len(inputs)

    input_str = ''
    for i, j, k, l in zip(env, inputs, output_dir, file):
        input_str += f'{i} {j} {k} {l} '

    sbatch(ALIGN_SBATCH_FILE, ALIGN_MEM, ALIGN_JOBS_NAME, len(inputs), input_str)

    wait_until_jobs_finished(ALIGN_FILE, len(inputs))

# def align_mtcnn(input_dir):
#    #import pdb;  pdb.set_trace()
#    print(f'Start align: {input_dir}')
#    output_dir = aligned_output_dir(input_dir)
#    os.system(f'{START_MTCNN_ENV} python {ALIGN_FUNC} --input-dir {input_dir} --output-dir {output_dir}')
#    print(f'End align: {input_dir}')


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
