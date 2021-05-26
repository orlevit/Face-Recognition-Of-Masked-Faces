#ODO: 2.change forehead image
# TODO: 3.create corona mask
#ODO: 4. obly get one identity out of the whole image
#ODO: 5. add saving to the proper location of image mask
#ODO: 6. refactor main.py
#ODO: 7. change render function
#change to a folder with images, or another list containing image paths

import os
import cv2
from tqdm import tqdm
from helpers import get_model, save_image, get_1id_pose, read_images, color_face_mask
from create_masks import create_masks, bg_color, render
from config import MASKS_NAMES

# Get img2pose model
model, transform = get_model()

# Get the masks and their complement
masks, masks_add, rest_of_heads = create_masks()

# Paths of all the images to create masks for
img_paths = read_images()

for img_path in tqdm(img_paths):
    # Read  image
    img = cv2.imread(img_path, 1)
    image_name = os.path.split(img_path)[1]

    # results from img2pose
    results = model.predict([transform(img)])[0]

    # Get only one 6DOF from all the 6DFs that img2pose found
    pose = get_1id_pose(results, img)

    print(img_path, '\n', pose)

    for mask, mask_add, rest_of_head, mask_name in zip(masks, masks_add, rest_of_heads, MASKS_NAMES):
        # Get the location of the masks on the image
        mask_x, mask_y, rest_mask_x, rest_mask_y = render(img.copy(), pose, mask, mask_add, rest_of_head, mask_name)

        # The average color of the surrounding of the image
        color = bg_color(mask_x, mask_y, img)

        # Put the colored mask on the face in the image
        masked_image = color_face_mask(img, color, mask_x, mask_y, rest_mask_x, rest_mask_y, mask_name)

        # Save masked image
        save_image(img_path, mask_name, masked_image)
