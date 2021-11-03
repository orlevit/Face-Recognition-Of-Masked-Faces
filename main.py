from tqdm import tqdm
from create_masks import masks_templates, process_image
from helpers import parse_arguments, get_model, read_images


def main(args):
    # Get the masks and their complement
    masks_to_create = masks_templates(args.masks)

    # Get img2pose model
    model, transform = get_model()

    # Paths of all the images to create masks for
    img_paths = read_images(args.input, args.image_extensions)

    # Process each image and create the requested masks
    for img_path in tqdm(img_paths):
        process_image(img_path, model, transform, masks_to_create, args)


if __name__ == '__main__':
    # Get the input and output directories and create the masks
    args = parse_arguments()
    main(args)
