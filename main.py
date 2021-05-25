# TODO: 
# 1.Save ti images
# 2.change forehead image
# 3.create corona mask
# 4. obly get one identity out of the whole image
#change to a folder with images, or another list containing image paths
from config import EYE_MASK_IND, HAT_MASK_IND, SCARF_MASK_IND, EYE_MASK, HAT_MASK, SCARF_MASK, MASKS_NAMES, IMAGES_PATH, THRESHOLD

if os.path.isfile(IMAGES_PATH):
    img_paths = pd.read_csv(images_path, delimiter=" ", header=None)
    img_paths = np.asarray(img_paths).squeeze()
else:
    img_paths = [os.path.join(images_path, img_path) for img_path in os.listdir(images_path)]

for img_path in tqdm(img_paths):
#     img = Image.open(img_path).convert("RGB")
    img = cv2.imread(img_path, 1)
    image_name = os.path.split(img_path)[1]
    
    res = img2pose_model.predict([transform(img)])[0]

    all_bboxes = res["boxes"].cpu().numpy().astype('float')

    poses = []
    bboxes = []
    for i in range(len(all_bboxes)):
        if res["scores"][i] > THRESHOLD:
            bbox = all_bboxes[i]
            pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
            pose_pred = pose_pred.squeeze()

            poses.append(pose_pred)  
            bboxes.append(bbox)

            print(img_path,'\n',poses)
    for mask, rest_of_head, mask_name in zip(masks, rest_of_heads, MASKS_NAMES):
        mask_x, mask_y, rest_mask_x, rest_mask_y = render(img.copy(), poses, mask, rest_of_head, mask_name)
        
        color = bg_color(mask_x, mask_y, img)
        img_output = img.copy()
        for x, y in zip(mask_x, mask_y):
            img_output[round(y), round(x), :] = [color[0], color[1], color[2]]  # BGR
        for x, y in zip(rest_mask_x, rest_mask_y):
            img_output[round(y), round(x), :] = img[round(y), round(x), :]
