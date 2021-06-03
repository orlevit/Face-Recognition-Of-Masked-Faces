import argparse
import multiprocessing


def run_multy(func, inputs):
    jobs = []
    for input in enumerate(inputs):
        # print(i)
        p = multiprocessing.Process(target=func, args=(input,))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Directory with input masked faces')
    # parser.add_argument('-o', '--output', type=str, help='Output directory.')
    # parser.add_argument('-e', '--image_extensions', default='.jpg,.bmp,.jpeg,.png',
    #                     type=str, help='The extensions of the images.')
    # parser.add_argument('-m', '--masks', default=ALL_MASKS, type=str, help='Which masks to create.')
    # parser.add_argument('-t', '--threshold', default=0.0, type=float,
    #                     help='The minimum confidence score for img2pose for face detection')

    return parser.parse_args()


