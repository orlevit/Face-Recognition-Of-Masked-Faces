import matplotlib.pyplot as plt
from config_file import SAME_AREA_DIST, THRESHOLD_BUFFER
import numpy  as np
from create_masks import  neighbors_cells_z, clustering, threshold_front
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from helpers import mark_image_with_mask
import heapq

###################################### calculate clusterings ################################

def otsu_clustering(elements):
    threshold = threshold_otsu(elements)
    cluster1_arr = []
    cluster2_arr = []
    equal_threshold = []
    for i in elements:
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
    cluster1_std = int(round(np.std(cluster1_arr), 0))
    cluster2_std = int(round(np.std(cluster2_arr), 0))

    return cluster1, cluster2, cluster1_std, cluster2_std


def clustering(elements_list):
    a=0
    b=0
    elements = np.asarray(elements_list)
    cluster1, cluster2, otsu_cluster1_std, otsu_cluster2_std = otsu_clustering(elements)
    clusters_std = np.array([otsu_cluster1_std, otsu_cluster2_std])

    if (clusters_std >= STD_CHECK).any():
        a+=1
        cluster1, cluster2 = kmean_clustering(elements, clusters_std)
    else:
        b+=1
    return cluster1, cluster2,a,b


def kmean_clustering(elements, clusters_std):
    cluster_number = 3

    while (clusters_std >= STD_CHECK).any():
        if cluster_number == 4:
            aaa=1
            pass
        # kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(elements[:, None])
        kmeans = KMeans(n_clusters=cluster_number, n_init=1, max_iter=100, random_state=0, tol=0.5).fit(elements[:, None])
        clusters_std = np.asarray([np.std(elements[kmeans.labels_ == i]) for i in range(cluster_number)])
        cluster_number += 1

    if cluster_number - 1 != 3 and cluster_number - 1 != 2:
        print("cluster_number: ",cluster_number-1)
    highest_clusters = np.argpartition(kmeans.cluster_centers_, -2, axis=0)[-2:]
    cluster1 = kmeans.cluster_centers_[highest_clusters[0]]
    cluster2 = kmeans.cluster_centers_[highest_clusters[1]]

    return cluster1, cluster2
################################################################################################
###################### points stds on the images
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
            cluster1, cluster2, cluster1_std, cluster2_std, thr= clustering2(surrounding_mask)
            diff = abs(cluster1 - cluster2)
            if (x == X_VALUE) and (y == Y_VALUE):  ############################################################## print
                a = list(surrounding_mask)
                clusters = [ cluster1, cluster2]
                min_cluster = min(cluster1, cluster2)
                threshold_buffer = diff * THRESHOLD_BUFFER
                threshold = min_cluster + threshold_buffer
                clusters_y = [5] * len(clusters)
                plt.figure(f'z: {z}, threshold:{thr}, cluster1:({np.round(cluster1)}+-{cluster1_std}), cluster2({np.round(cluster2)}+-{cluster2_std})')
                plt.hist(a,bins=100)
                plt.xticks(range(700, 950, 10), range(700, 950, 10))
                plt.vlines(clusters, ymin=0, ymax=10, color='k')
                plt.vlines(threshold, ymin=0, ymax=10, color='y')
                plt.vlines(thr, ymin=0, ymax=10, color='r')
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

    return cluster1, cluster2, int(round(np.std(cluster1_arr),0)), int(round(np.std(cluster2_arr),0)), threshold

def plt_viz(r_img, df,X_VALUE, Y_VALUE):
    unique_df = df.sort_values(['z', 'mask'], ascending=False).drop_duplicates(['x', 'y', 'mask'], keep='first')
    frontal_add_mask_with_bg = unique_df[(unique_df['mask'] == 2)][['x', 'y', 'z']]

    mask_ind = threshold_front(r_img, df, frontal_add_mask_with_bg)
    add_mask_on_image2 = mark_image_with_mask(mask_ind,r_img, 1)
    y_front, x_front = np.where(add_mask_on_image2 == 1)
    plt.figure()
    plt.scatter(X_VALUE, -Y_VALUE, color='g', s=130)
    plt.annotate(f'{X_VALUE}, {Y_VALUE}', (X_VALUE, -Y_VALUE))
    plt.scatter(x_front, -y_front, s=20, c='r')
    plt.scatter(frontal_add_mask_with_bg.x, -frontal_add_mask_with_bg.y, s=5)
    plt.show()



def plt_stds(r_img, df, STD_CHECK):
    unique_df = df.sort_values(['z', 'mask'], ascending=False).drop_duplicates(['x', 'y', 'mask'], keep='first')
    frontal_add_mask_with_bg = unique_df[(unique_df['mask'] == 2)][['x', 'y', 'z']]

    mask_on_img_front, mask_on_img_back, mask_on_img_less_diff, std_front_threshold, std_back_threshold, std_less_diff, above_std_check =\
        threshold_front_with_std(r_img, df, frontal_add_mask_with_bg, STD_CHECK, KMEANS_IND=True)
    masks_on_names = ["mask_on_img_front", "mask_on_img_back", "mask_on_img_less_diff"]
    masks_on = [mask_on_img_front, mask_on_img_back, mask_on_img_less_diff]
    print(f'above std {STD_CHECK}:   {above_std_check}')
    for i, (mask_on, std) in enumerate(zip(masks_on, [ std_front_threshold, std_back_threshold, std_less_diff])):
        if mask_on.shape[-1] == 2:
            plt.figure(masks_on_names[i])
            plt.scatter(mask_on[:, 1], -mask_on[:, 0], color='g', s=1)
            for i, txt in enumerate(std):
                plt.annotate(f'{txt}', (mask_on[i, 1], -mask_on[i, 0]))
        else:
            print(f'No points for: {masks_on_names[i]}')

        plt.show()



def threshold_front_with_std(r_img, df,frontal_mask_all, STD_CHECK=-99999999, KMEANS_IND=False):
    def in_out(surrounding_mask):
        cluster1, cluster2, cluster1_std, cluster2_std, thr = clustering2(surrounding_mask)
        if (cluster1_std < STD_CHECK) and (cluster2_std <STD_CHECK):
            return cluster1, cluster2, cluster1_std, cluster2_std, thr,  False, 0

        kmeans_iters = 0
        cluster_number = 3
        clusters_std = np.array([cluster1_std,cluster2_std])
        surrounding_mask = np.asarray(surrounding_mask)
        while (clusters_std>=STD_CHECK).any():
            kmeans = KMeans(n_clusters=cluster_number, random_state=0).fit(np.asarray(surrounding_mask)[:,None])
            clusters_std = np.asarray([np.std(surrounding_mask[kmeans.labels_ == i]) for i in range(cluster_number)])
            cluster_number += 1
            kmeans_iters += 1


        hci = np.argpartition(kmeans.cluster_centers_, -2,axis=0)[-2:]
        stds = [np.std(surrounding_mask[kmeans.labels_ == i]) for i in hci]

        return kmeans.cluster_centers_[hci[0]], kmeans.cluster_centers_[hci[1]], int(np.round(stds[0])), int(np.round(stds[1])), 000, True, kmeans_iters

    img_x_dim, img_y_dim = r_img.shape[1], r_img.shape[0]
    mask_on_img = np.asarray([[None] * img_x_dim] * img_y_dim)
    mask_on_img_front = []
    mask_on_img_back = []
    mask_on_img_less_diff = []

    std_front_threshold = []
    std_back_threshold = []
    std_less_diff = []
    above_std_check = 0
    SAME_AREA_IND = False
    # Each pixel contains a list of all the Z coordinates from the 3D model
    for x, y, z in zip(df.x, df.y, df.z):
        if isinstance(mask_on_img[y, x], type(None)):
            mask_on_img[y, x] = [z]
        else:
            mask_on_img[y, x].append(z)

    for x, y, z in zip(frontal_mask_all.x, frontal_mask_all.y, frontal_mask_all.z):
        surrounding_mask = neighbors_cells_z(mask_on_img, x, y, img_x_dim - 1, img_y_dim - 1)

        if len(np.unique(surrounding_mask)) not in [0, 1]:
            if KMEANS_IND:
                cluster1, cluster2, cluster1_std, cluster2_std, thr, SAME_AREA_IND, kmeans_iters = in_out(surrounding_mask) # clustering2(surrounding_mask)
                max_cluster_std = f'{str(max(cluster1_std, cluster2_std))}-{kmeans_iters}'

            else:
                cluster1, cluster2, cluster1_std, cluster2_std, thr =  clustering2(surrounding_mask)
                max_cluster_std = str(max(cluster1_std, cluster2_std))


            if  KMEANS_IND or (cluster1_std>STD_CHECK or cluster2_std> STD_CHECK):
                above_std_check+=1
                diff = abs(cluster1 - cluster2)
                if SAME_AREA_IND or SAME_AREA_DIST < diff:
                    min_cluster = min(cluster1, cluster2)

                    threshold_buffer = diff * THRESHOLD_BUFFER
                    threshold = min_cluster + threshold_buffer
                    if z < threshold:
                        std_back_threshold.append(max_cluster_std)
                        mask_on_img_back.append((y,x))
                    else:
                        std_front_threshold.append(max_cluster_std)
                        mask_on_img_front.append((y,x))
                else:
                    std_less_diff.append(max_cluster_std)
                    mask_on_img_less_diff.append((y,x))
        else:
            std_front_threshold.append('unique')
            mask_on_img_front.append((y,x))

    mask_on_img_front = np.asarray(mask_on_img_front)
    mask_on_img_back = np.asarray(mask_on_img_back)
    mask_on_img_less_diff = np.asarray(mask_on_img_less_diff)

    return mask_on_img_front, mask_on_img_back, mask_on_img_less_diff, std_front_threshold, std_back_threshold, std_less_diff, above_std_check

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