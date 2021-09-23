import matplotlib.pyplot as plt
from config_file import SAME_AREA_DIST, THRESHOLD_BUFFER
import numpy  as np
from create_masks import  neighbors_cells_z, clustering, threshold_front
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt
from helpers import mark_image_with_mask

def dist(r_img, df,X_VALUE,Y_VALUE):
    unique_df = df.sort_values(['z', 'mask'], ascending=False).drop_duplicates(['x', 'y', 'mask'], keep='first')
    frontal_mask_all = unique_df[(unique_df['mask'] == 2)][['x', 'y', 'z']]

    img_x_dim, img_y_dim = r_img.shape[1], r_img.shape[0]
    mask_on_img = np.asarray([[None] * img_x_dim] * img_y_dim)
    mask_on_img_front = np.zeros((img_y_dim, img_x_dim))

    # Each pixel contains a list of all the Z coordinates from the 3D model
    for x, y, z in zip(df.x, df.y, df.z):
        if isinstance(mask_on_img[y, x], type(None)):
            mask_on_img[y, x] = [z]
        else:
            mask_on_img[y, x].append(z)

    for x, y, z in zip(frontal_mask_all.x, frontal_mask_all.y, frontal_mask_all.z):
        surrounding_mask = neighbors_cells_z(mask_on_img, x, y, img_x_dim - 1, img_y_dim - 1)
        mask_on_img_front[y, x] = 1

        if len(np.unique(surrounding_mask)) not in [0, 1]:
            cluster1, cluster2, cluster1_std, cluster2_std= clustering2(surrounding_mask)
            diff = abs(cluster1 - cluster2)
            if (x == X_VALUE) and (y == Y_VALUE):  ############################################################## print
                a = list(surrounding_mask)
                clusters = [ cluster1, cluster2]
                min_cluster = min(cluster1, cluster2)
                threshold_buffer = diff * THRESHOLD_BUFFER
                threshold = min_cluster + threshold_buffer
                clusters_y = [5] * len(clusters)
                plt.figure(f'z: {z}, cluster1_std:{cluster1_std}, cluster2_std:{cluster2_std}')
                plt.hist(a,bins=100)
                plt.xticks(range(700, 950, 10), range(700, 950, 10))
                plt.vlines(clusters, ymin=0, ymax=10, color='k')
                plt.vlines(threshold, ymin=0, ymax=10, color='y')
                plt.vlines([z], ymin=0, ymax=10, color='c')
                plt.show()
                break

def clustering2(elements_list):
    threshold = threshold_otsu(np.array(elements_list))
    cluster1_arr = []
    cluster2_arr = []
    equal_threshold = []
    for i in elements_list:
        if i < threshold:
            cluster1_arr.append(i)
        elif i > threshold:
            cluster2_arr.append(i)
        else:
            equal_threshold.append(i)

    if not cluster1_arr:
        cluster1_arr.extend(equal_threshold)
    else:
        cluster2_arr.extend(equal_threshold)

    cluster1 = np.mean(cluster1_arr)
    cluster2 = np.mean(cluster2_arr)

    return cluster1, cluster2, round(np.std(cluster1_arr),1), round(np.std(cluster2_arr),1)

def plt_viz(r_img, df,X_VALUE, Y_VALUE):
    unique_df = df.sort_values(['z', 'mask'], ascending=False).drop_duplicates(['x', 'y', 'mask'], keep='first')
    frontal_add_mask_with_bg = unique_df[(unique_df['mask'] == 2)][['x', 'y', 'z']]
    mask_ind = threshold_front(r_img, df, frontal_add_mask_with_bg)
    add_mask_on_image2 = mark_image_with_mask(mask_ind,r_img, 1)
    y_front, x_front = np.where(add_mask_on_image2 == 1)
    plt.scatter(X_VALUE, -Y_VALUE, color='g', s=130)
    plt.annotate(f'{X_VALUE}, {Y_VALUE}', (X_VALUE, -Y_VALUE))
    plt.scatter(x_front, -y_front, s=20, c='r')
    plt.scatter(frontal_add_mask_with_bg.x, -frontal_add_mask_with_bg.y, s=5)
    plt.show()

##############################################################################################################
# from skimage.filters import threshold_otsu
# txt1 = "684 686 685 933 931 683 932 682 681 664 679 682 662 667 669 672 677 679 681 666 676 660 664 670 668 670 668 679 666 672 676 675 678 678 677 660 675 674 673 933 931 657 670 672 672 671 674 674 673 673 676 932"
# txt2 = "931 683 932 682 681 664 680 663 679 682 662 667 669 672 677 679 681 666 676 660 664 670 668 670 668 679 666 672 676 675 678 933 931 657 670 672 672 671 674 674 673 673 676 932 656 660 933"
# txt3 = "686 685 933 687 683 932 684 670 684 680 663 679 682 662 667 669 672 677 934 672 676 675 679 681 935 666 678 681 678 677 660 675 674 673 933"
# for txt in [txt1, txt2, txt3]:
#     ts = txt.split()
#     arr = np.array([int(i) for i in ts])
#     print(f'----------------------{len(arr)}----------------------')
#     # arr = np.random.randint(100, size=100)
#     tic = time()
#     threshold = threshold_otsu(arr)
#     cluster1 = np.mean(arr[arr < threshold])
#     cluster2 = np.mean(arr[arr > threshold])
#     toc = time()
#     print(f'{cluster1}, {cluster2}: {toc - tic}')
#     tic = time()
#     km = KMeans(n_clusters=2, random_state=0, max_iter=10, tol=1, n_init=1).fit(arr[:, None])
#     toc = time()
#     print(f'{np.squeeze(km.cluster_centers_)}: {toc - tic}')
#     print('-----------------------')
##############################################################################################################