############# unused in final version
import numpy as np
#############
from tqdm import tqdm
from create_masks import masks_templates, process_image
from helpers import parse_arguments, get_model, read_images
from line_profiler_pycharm import profile


# TODO: 0) check morphologicals sunglasses and eye mask
# TODO: 1)run big examples
# TODO: 2) profiling the code + refactoring
# TODO: 3)run ALL masks
# TODO: 4)run in multithreads + check outputs to stdout of the main program
# TODO: 5)add option for run in multithreads
# TODO: 7) refactoring the code and profiling
# TODO: 8) add documentation

@profile
def main(args):
    time_total = []

    # Get the masks and their complement
    masks_to_create = masks_templates(args.masks)

    # Get img2pose model
    model, transform = get_model()

    # Paths of all the images to create masks for
    img_paths = read_images(args.input, args.image_extensions)

    # Process each image and create the requested masks
    i=0
    for img_path in tqdm(img_paths):
    # for img_path in img_paths:
        time1,ii = process_image(img_path, model, transform, masks_to_create, args)
        time_total.append(time1)
        i+=ii
        if i > 50:
            break
    print(np.mean(time_total))


if __name__ == '__main__':
    # Get the input and output directories and create the masks
    args = parse_arguments()
    main(args)
