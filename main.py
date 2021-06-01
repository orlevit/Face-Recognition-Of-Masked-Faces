import cv2
from tqdm import tqdm
from torch.multiprocessing import spawn

from config_file import config
from create_masks import create_masks, bg_color, render
from helpers import get_model, save_image, get_1id_pose, read_images, color_face_mask, parse_arguments

from time import time

# todo add sunglasses mask
# todo split masks ind to another file & change create masks name
# todo run the program
# todo test times i\o bound or cpu bound
# todo test gpu time
# todo multitread with ? multi i/o?

def run_masks(ps_idx, img_path, masks_to_create, model, transform, args, config):
    io_time = 0
    cpu_time = 0
    img2pose_time = 0
    print('\n', img_path)
    # Read an image
    t1 = time()
    img = cv2.imread(img_path, 1)
    io_time_curr1 = time() - t1
    io_time += io_time_curr1
    # results from img2pose
    t_img2pose1 = time()
    results = model.predict([transform(img)])[0]
    img2pose_time_curr = time() - t_img2pose1
    img2pose_time += img2pose_time_curr
    # Get only one 6DOF from all the 6DFs that img2pose found
    pose = get_1id_pose(results, img, args.threshold)

    # face detected with img2pose and above the threshold
    if pose.size != 0:
        # for mask, mask_add, rest_of_head, mask_name in zip(masks, masks_add, rest_of_heads, MASKS_NAMES):
        for mask_name in masks_to_create:
            # Get the location of the masks on the image
            mask_x, mask_y, rest_mask_x, rest_mask_y = render(img, pose, mask_name, config)

            # The average color of the surrounding of the image
            color = bg_color(mask_x, mask_y, img,config)

            # Put the colored mask on the face in the image
            masked_image = color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name, config)

            # Save masked image
            twrite = save_image(img_path, mask_name, masked_image, args.output)
            io_time_curr2 = twrite
            io_time += io_time_curr2
    else:
        print(f'No face detected for: {img_path}')
    cpu_time += time() - t1 - img2pose_time_curr - io_time_curr1 - io_time_curr2
    print(f'io_time: {io_time}')
    print(f'img2pose_time: {img2pose_time}')
    print(f'cpu_time: {cpu_time}')


def main(args):
    t1=time()

    # Get the masks and their complement
    masks_to_create = create_masks(args.masks,config)

    # Get img2pose model
    model, transform = get_model()

    # Paths of all the images to create masks for
    img_paths = read_images(args.input, args.image_extensions)

    #thread_args = [[img_path] + [masks_to_create, model, transform] for img_path in img_paths]
    for img_path in img_paths:
        spawn(run_masks,args=(img_path, masks_to_create, model, transform, args, config))
        #p.start()
        #del p
    print('time in total: ',time()-t1)
#    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
#        pool.starmap(run, thread_args)


if __name__ == '__main__':
    # Get the input and output directories and create the masks
    args = parse_arguments()
    main(args)
