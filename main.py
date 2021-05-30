import cv2
from tqdm import tqdm
from config_file import config
from create_masks import create_masks, bg_color, render
from helpers import get_model, save_image, get_1id_pose, read_images, color_face_mask, parse_arguments


# todo add sunglasses mask
# todo split masks ind to another file & change create masks name
# todo run the program
def main(args):
    # Get the masks and their complement
    masks_to_create = create_masks(args.masks)

    # Get img2pose model
    model, transform = get_model()

    # Paths of all the images to create masks for
    img_paths = read_images(args.input, args.image_extensions)

    for img_path in tqdm(img_paths):
        # Read an image
        img = cv2.imread(img_path, 1)

        # results from img2pose
        results = model.predict([transform(img)])[0]

        # Get only one 6DOF from all the 6DFs that img2pose found
        pose = get_1id_pose(results, img)

        # for mask, mask_add, rest_of_head, mask_name in zip(masks, masks_add, rest_of_heads, MASKS_NAMES):
        for mask_name in masks_to_create:
            # Get the location of the masks on the image
            mask_x, mask_y, rest_mask_x, rest_mask_y = render(img, pose, mask_name)

            # The average color of the surrounding of the image
            color = bg_color(mask_x, mask_y, img)

            # Put the colored mask on the face in the image
            masked_image = color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name, config)

            # Save masked image
            save_image(img_path, mask_name, masked_image, args.output)


if __name__ == '__main__':
    # Get the input and output directories and create the masks
    args = parse_arguments()
    main(args)
