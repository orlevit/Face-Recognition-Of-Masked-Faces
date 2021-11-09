# Check all the IDs in the pairs, to detect how many distinct images there are.
import os
import random

# Locations of the files 
loc='/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/files'
images_loc = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/images_masked_crop/agedb30/{}/'
file_pairs = os.path.join(loc, 'original', 'agedb30_pairs.txt')
mask_type = 'covid19mask'#, 'eyemask', 'hatmask', 'sunglassesmask']
IMAGES_REVIEW = 10000000
def add_create(id_dict, id, img_number):
    img_number = int(img_number)
    if id in id_dict:
        id_dict[id].add(img_number)
    else:
        id_dict[id] = set([img_number])

    return id_dict

pairs_dict = {}
first_line = True

with open(file_pairs) as f:
    for one_line in f:
        l = one_line.strip().split('\t')
        if first_line:
            first_line = False
            continue
        if len(l) == 3:
            pairs_dict = add_create(pairs_dict, l[0], l[1])
            pairs_dict = add_create(pairs_dict, l[0], l[2])
        elif len(l) == 4:
            pairs_dict = add_create(pairs_dict, l[0], l[1])
            pairs_dict = add_create(pairs_dict, l[2], l[3])


pairs_list = list(pairs_dict)
unique_imgs = []
item_num = 0
random.shuffle(pairs_list)
pairs_list_iter = iter(pairs_list)
while pairs_list_iter.__length_hint__():
      person_id = next(pairs_list_iter)
      person_id_nums = iter(pairs_dict[person_id])
      while person_id_nums.__length_hint__() and item_num < IMAGES_REVIEW:
            person_id_num = next(person_id_nums)
            unique_imgs.append('1')#images_loc.format(mask_type) + person_id + '/'+person_id+'_'+'%04d'%person_id_num+'.jpg')
            item_num += 1

print(len(unique_imgs))
#for img in unique_imgs:
#    print(img)
