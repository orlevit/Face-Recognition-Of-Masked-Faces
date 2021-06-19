from scripts.script_helper import sbatch, delete_create_file, wait_until_jobs_finished
from scripts.script_config import TEST_MEM, TEST_JOBS_NAME, TEST_SBATCH_FILE, TEST_FILE, ARCFACE_ENV

def make_test(inputs):
    delete_create_file(TEST_FILE)
    # env = [ARCFACE_ENV] * len(inputs)
    # file = [TEST_FILE] * len(inputs)

    input_str = ''
    for i, j, k, l in zip(env, inputs, output_dir, file):
       input_str += f'{i} {j} {k} {l} '

    sbatch(TEST_SBATCH_FILE, TEST_MEM, TEST_JOBS_NAME, len(inputs), input_str)

    wait_until_jobs_finished(TEST_FILE, len(inputs))

if __name__ == '__main__':
    # args = get_args()
    make_test(inputs)
