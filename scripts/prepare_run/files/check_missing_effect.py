# Checking the effect f the missings images and id's on the pairs file and the lst file
# make sure that the first line in the log missing:"file_missings_log" is not important line(it is ignored in the code due to "first_line=True"
import os

# Locations of the files 
loc='/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/files'
file_lst_input = os.path.join(loc, 'original', 'casia.lst')
file_lst_output = os.path.join(loc,'no_missings' ,'casia_no_missings.lst')
file_pairs = os.path.join(loc, 'original', 'casia_pairs_5k.txt')
file_missings_log = os.path.join(loc, 'log_casia_missing.txt') # the missing log from masks creation

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
            
first_line = True
missings_dict = {}

with open(file_missings_log) as f:
    for one_line in f:
        l = one_line.strip().split('/')[-2:]
        if first_line:
            first_line = False
            continue
        missings_dict = add_create(missings_dict, l[0], l[1].split('.')[0])

for k1, v1 in missings_dict.items():
    for k2, v2 in pairs_dict.items():
        if k1 == k2 and v1&v2: 
            print(k1, v1&v2)      
            
id_orginal_count = set()
id_clean_missings_count = set()
first_line = True

with open(file_lst_input) as f1, open(file_lst_output, 'w') as f2:
    for one_line in f1:
        l = one_line.strip().split('/')[-2:]
        id = l[0]; img_number = l[1].split('.')[0]
        id_orginal_count.add(int(id))
        
        if id in missings_dict.keys() and int(img_number) in missings_dict[id]:
            continue

        f2.write(one_line)
        id_clean_missings_count.add(int(id))
if len(id_clean_missings_count&id_orginal_count) != len(id_orginal_count):
    print(f'There is missing id due to removing of all the images of the same id: {len(id_clean_missings_count&id_orginal_count)} and {len(id_orginal_count)}') 
