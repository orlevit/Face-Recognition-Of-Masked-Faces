############# unused in final version
from time import time
import numpy as np
#############
import cv2
from tqdm import tqdm
from config_file import config
from create_masks import masks_templates, bg_color, render
from helpers import get_model, save_image, get_1id_pose, read_images, color_face_mask, parse_arguments, resize_image
from line_profiler_pycharm import profile

# TODO: 0.1) what to do with more than 3 cluters
# TODO: 0.2) check again the sunglasses, if the back glass is shown
# TODO: 1)run small exaples
# TODO: 2) profiling the code
# TODO: 3)run ALL masks
# TODO: 4)run in multithreads
# TODO: 5)add option for run in multithreads
# TODO: 6)change coronamask to covid19 mask
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
    for img_path in tqdm(img_paths):
        # Read an image
        img = cv2.imread(img_path, 1)

        # results from img2pose
        results = model.predict([transform(img)])[0]
        tic =time()

        # Get only one 6DOF from all the 6DFs that img2pose found
        pose, bbox = get_1id_pose(results, img, args.threshold)

        # face detected with img2pose and above the threshold
        if pose.size != 0:
            # Resize image that ROI will be in a fix size
            r_img, scale_factor = resize_image(img, bbox)

            # for mask, mask_add, rest_of_head, mask_name in zip(masks, masks_add, rest_of_heads, MASKS_NAMES):
            for mask_name in masks_to_create:
                # Get the location of the masks on the image
                mask_x, mask_y, rest_mask_x, rest_mask_y = render(img, r_img, pose, mask_name, scale_factor)

                # The average color of the surrounding of the image
                color = bg_color(mask_x, mask_y, img)

                # Put the colored mask on the face in the image
                masked_image = color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name, config)

                # Save masked image
                save_image(img_path, mask_name, masked_image, args.output, bbox, args.bbox_ind, args.inc_bbox)
        else:
            print(f'No face detected for: {img_path}')
        toc =time()
        time_total.append(toc-tic)
    print(np.mean(time_total))
if __name__ == '__main__':
    # Get the input and output directories and create the masks
    args = parse_arguments()
    main(args)
