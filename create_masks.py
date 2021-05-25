# todo: delete render_plot

import cv2
import numpy as np
from config import VERTICES_PATH, EYE_MASK_IND, HAT_MASK_IND, SCARF_MASK_IND, EYE_MASK, HAT_MASK, SCARF_MASK
from project_on_image import transform_vertices
# This is the opposite of the functon in expression-net-old
def get_hat_mask_index(a1,b1,c1,x_left,x_right,x,y):

	index_list = []
	for i in range(len(x)):
	  if ((y[i] > (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and
              (x[i] > x_left) and (x[i] < x_right)):# or (x[i] <= x_left) or (x[i] >= x_right):

	      index_list.append(i)

	return(index_list)

# This is the opposite of the functon in expression-net-old
# get the scarf mask indexes on the model
def get_scarf_mask_index(a1,b1,c1,x_left,x_right,x,y):

	index_list = []
	for i in range(len(x)):
		if ((y[i] < (a1 * x[i] ** 2 + b1 * x[i] + c1)) and
              (x[i] > x_left) and (x[i] < x_right)):# or (x[i] <= x_left) or (x[i] >= x_right):

			index_list.append(i)

	return(index_list)

# This is the opposite of the functon in expression-net-old
def get_eyes_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y):
    index_list = []
    for i in range(len(x)):
        if ((y[i] < (a1 * (x[i] ** 2) + b1 * x[i] + c1)) and
                (y[i] > (a2 * x[i] ** 2 + b2 * x[i] + c2)) and
                (x[i] > x_left - 2) and (x[i] < x_right + 2)):
            index_list.append(i)

    index_list = np.setdiff1d(range(len(x)), index_list)
    return (index_list)

def MakeEyesMask(model, SEP, x, y, z):
    x_left = x[EYE_MASK_IND[0]]
    x_right = x[EYE_MASK_IND[1]]
    y_left = y[EYE_MASK_IND[0]]
    y_right = y[EYE_MASK_IND[1]]
    x_middle = x[EYE_MASK_IND[2]]
    y_choosen_top = y[EYE_MASK_IND[2]]
    y_choosen_down = y[EYE_MASK_IND[3]]

    x_3points = [x_left, x_middle, x_right]
    y_3points1 = [y_left, y_choosen_down, y_right]
    y_3points2 = [y_left, y_choosen_top, y_right]

    a1, b1, c1 = np.polyfit(x_3points, y_3points2, 2)
    a2, b2, c2 = np.polyfit(x_3points, y_3points1, 2)

    index_list = get_eyes_mask_index(a1, b1, c1, a2, b2, c2, x_left, x_right, x, y)

    return (index_list)


def MakeScarfMask(model, SEP, x, y, z):
    x_left = x[SCARF_MASK_IND[0]]
    x_right = x[SCARF_MASK_IND[1]]
    y_left = y[SCARF_MASK_IND[0]]
    y_right = y[SCARF_MASK_IND[1]]
    x_middle = x[SCARF_MASK_IND[2]]
    y_choosen_top = y[SCARF_MASK_IND[2]]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_choosen_top, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_scarf_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return (index_list)


# create hat mask
def MakeHatMask(model, SEP, x, y, z):
    x_left = x[HAT_MASK_IND[0]]
    x_right = x[HAT_MASK_IND[1]]
    y_left = y[HAT_MASK_IND[0]]
    y_right = y[HAT_MASK_IND[1]]
    x_middle = x[HAT_MASK_IND[2]]
    y_choosen_down = y[HAT_MASK_IND[2]]

    x_3points = [x_left, x_middle, x_right]
    y_3points = [y_left, y_choosen_down, y_right]
    a1, b1, c1 = np.polyfit(x_3points, y_3points, 2)

    index_list = get_hat_mask_index(a1, b1, c1, x_left, x_right, x, y)
    return (index_list)


def render(img, pose, mask_verts, rest_of_head_verts, mask_name):

    # Transform the 3DMM according to the pose
    mask_trans_vertices = transform_vertices(img, pose, mask_verts)
    rest_trans_vertices = transform_vertices(img, pose, rest_of_head_verts)
    
    # Whether to add the forehead to the mask, this is currenly only used for eye and hat masks
    if mask_name in [HAT_MASK, EYE_MASK]:
        mask_x, mask_y = add_headTop(img, mask_trans_vertices, rest_trans_vertices)
        mask_x, mask_y = np.append(mask_x.flatten(), mask_trans_vertices[:, 0]), \
                         np.append(mask_y.flatten(), mask_trans_vertices[:, 1])
        rest_x, rest_y = rest_trans_vertices[:, 0], rest_trans_vertices[:, 1]
    else:
        mask_x, mask_y = mask_trans_vertices[:, 0], mask_trans_vertices[:, 1]
        rest_x, rest_y = rest_trans_vertices[:, 0], rest_trans_vertices[:, 1]

    # Perform morphological close
    morph_mask_x, morph_mask_y = morphologicalClose(mask_x, mask_y, img)
    morph_rest_x, morph_rest_y = morphologicalClose(rest_x, rest_y, img)

    return morph_mask_x, morph_mask_y, morph_rest_x, morph_rest_y

def render_plot(x, y, img, bboxes):
    plt.figure(figsize=(8, 8))     
    plt.imshow(img)        
    plt.scatter(x,y)
    for bbox in bboxes:
        plt.gca().add_patch(patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=3,edgecolor='b',facecolor='none'))            
    plt.show()   

def indexOnSEP(index_list, SEP):
    x = SEP[:, 0];
    y = SEP[:, 1];
    z = SEP[:, 2];
    xMask = x[index_list];
    yMask = y[index_list];
    zMask = z[index_list];
    sizeArray = (xMask.shape[0], 3)
    maskSEP = np.zeros(sizeArray)
    maskSEP[:, 0] = xMask
    maskSEP[:, 1] = yMask
    maskSEP[:, 2] = zMask

    return (maskSEP)

def bg_color(mask_x, mask_y, image):
    morph_mask_x, morph_mask_y = morphologicalClose(mask_x, mask_y, image)
    # Get the average color of the whole mask
    image_bg = image.copy()
    image_bg_effective_size = image_bg.shape[0] * image_bg.shape[1] - len(mask_x)
    image_bg[morph_mask_x.astype(int),morph_mask_y.astype(int), :] = [0, 0, 0]
    image_bg_blue = image_bg[:, :, 0];
    image_bg_green = image_bg[:, :, 1];
    image_bg_red = image_bg[:, :, 2];
    image_bg_blue_val = np.sum(image_bg_blue) / image_bg_effective_size
    image_bg_green_val = np.sum(image_bg_green) / image_bg_effective_size
    image_bg_red_val = np.sum(image_bg_red) / image_bg_effective_size

    return [image_bg_blue_val, image_bg_green_val, image_bg_red_val]

def morphologicalClose(mask_x, mask_y, image):
    maskOnImage = np.zeros_like(image)
    for x, y in zip(mask_x, mask_y):
        if ((0 <= x <= maskOnImage.shape[0] - 1) and (0 <= y <= maskOnImage.shape[1] - 1)):
            maskOnImage[int(y), int(x), :] = [255, 255, 255]

    # morphology close
    gray_mask = cv2.cvtColor(maskOnImage, cv2.COLOR_BGR2GRAY)
    res, thresh_mask = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)  # kernerl filter
    morph_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_CLOSE, kernel)
    yy, xx = np.where(morph_mask == 255)

    return xx, yy

def add_headTop(image, mask_trans_vertices, rest_trans_vertices):
    maskOnImage = np.zeros_like(image)
    mask_x_ind, mask_y_ind = mask_trans_vertices[:, 0].astype(int), mask_trans_vertices[:, 1].astype(int)   
    for x, y in zip(mask_x_ind, mask_y_ind):
        maskOnImage[y, x] = 255

    bottom_hat = np.empty(image.shape[0])
    bottom_hat[:] = np.nan
    for i in range(image.shape[0]):
        iy, _ = np.where(maskOnImage[:, i])
        if iy != []:
            bottom_hat[i] = np.min(iy);

    print(mask_trans_vertices.shape, rest_trans_vertices.shape)
    all_face_proj = np.concatenate((mask_trans_vertices, rest_trans_vertices), axis = 0)
    print(all_face_proj.shape)
    all_face_proj_y = all_face_proj[:, 1]
    max, min = np.max(all_face_proj_y), np.min(all_face_proj_y)
    kernel_len = int((max - min) / 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, kernel_len))
    dilated_image = cv2.dilate(maskOnImage, kernel, iterations=1)

    forehead_x_ind, forehead_y_ind = [], []
    for i in range(image.shape[0]):
        iy, _ = np.where(dilated_image[:, i])
        jj = np.where(iy <= bottom_hat[i])
        if jj[0] != []:
            j = iy[jj]
            forehead_x_ind.append([i] * len(j))
            forehead_y_ind.append(j)

    return np.asarray(forehead_x_ind), np.asarray(forehead_y_ind)

def get_rest_mask(maskInd, verts):
    mask = indexOnSEP(maskInd, verts)
    rest_of_head_ind = np.setdiff1d(range(verts.shape[0]), maskInd)
    rest_of_head_mask = indexOnSEP(rest_of_head_ind, verts)
    
    return(rest_of_head_mask)

def load_3DMM():
	verts = np.load(VERTICES_PATH)
	th = 30
	R = [[1,0,0],[0,np.cos(th),-np.sin(th)],[0,np.sin(th),np.cos(th)]]
	verts_rotated = verts.copy()
	verts_rotated = np.matmul(verts_rotated, R)
	verts_rotated[:,2] *= -1
	return verts, verts_rotated

def create_masks():
        #TODO: 1.not to use verts_rotated but verts
#  todo: 2. chame Make**Mask funtions with out the None
	verts, verts_rotated = load_3DMM()
	x, y, z = verts_rotated[:,0],verts_rotated[:,1],verts_rotated[:,2]
	# Change the calling for these functions!
	eyeMaskInd = MakeEyesMask(None, None, x, z, None)
	hatMaskInd = MakeHatMask(None, None, x, z, None)
	scarfMaskInd = MakeScarfMask(None, None, x, z, None)

	masksInd = [eyeMaskInd, hatMaskInd, scarfMaskInd]

	masks = [indexOnSEP(maskInd, verts) for maskInd in masksInd]
	rest_of_heads = [get_rest_mask(maskInd, verts) for maskInd in masksInd]

	return masks, rest_of_heads
