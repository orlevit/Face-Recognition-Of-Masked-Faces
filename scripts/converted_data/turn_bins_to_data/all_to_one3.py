import os
import mxnet as mx
import torch
import numpy as np
from pprint import pprint
from datetime import datetime

#DATA_PATH = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000'
#TARGET_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000/all'
MASK = 'covid19'
DATA_PATH = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/a{MASK}mask/train/db_a{MASK}mask_a{MASK}mask'
TARGET_LOC = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/a{MASK}mask/train/db_a{MASK}mask_a{MASK}mask/{MASK}_model/all'
#DATA_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/tmp/test/db_nomask_lfw'
#TARGET_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/ready_data/350000_test_lfw_casia_pairs/nomask'
DATA_TARGET_LOC = os.path.join(TARGET_LOC, 'data.npy')
LABELS_TARGET_LOC = os.path.join(TARGET_LOC, 'labels.npy')

def get_files_path(data_path): 
    print(data_path)
    data_path_list = []
    labels_path_list = []
    
    for cur_dir1, dirs1, _ in os.walk(data_path):
        for sub_dir in dirs1:
            for cur_dir2, _, files in os.walk(os.path.join(cur_dir1, sub_dir)):
                for file in files:
                    path = os.path.join(cur_dir2,file)
                    name_of_file = path.rsplit('/',1)[-1] 
                    
                    if name_of_file.startswith('data'): 
                        data_path_list.append(path)
                    if name_of_file.startswith('labels'): 
                        labels_path_list.append(path)
                   
    data_path_list = sorted(data_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    labels_path_list = sorted(labels_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    pprint(data_path_list)
    pprint(labels_path_list)
    return data_path_list, labels_path_list

def compose_data(paths, is_data):
    all_data = None
    for path in paths:
        print('Processing file: ', path)
        tic = datetime.now()
        loaded = mx.nd.load(path)
        loaded_numpy = loaded[0].asnumpy()
        loaded_numpy = np.expand_dims(loaded_numpy, axis=0)
#        if is_data:
#           #import pdb;pdb.set_trace();
#           loaded_numpy = np.expand_dims(np.load(path, allow_pickle=True), axis=0)
#        else:
#           loaded_numpy = np.expand_dims(np.load(path, allow_pickle=True), axis=0)
        if all_data is None:
            all_data = loaded_numpy
        else:
            all_data = np.concatenate((all_data, loaded_numpy), axis=0)
            
        del loaded_numpy
        toc = datetime.now()
        print('Time: ',toc-tic)
        print(all_data.shape)
    return all_data 

data_path_list, labels_path_list =  get_files_path(DATA_PATH)
data = compose_data(data_path_list, True)
lables = compose_data(labels_path_list, False)

os.makedirs(TARGET_LOC)
torch.save(data, DATA_TARGET_LOC, pickle_protocol=4)
torch.save(lables, LABELS_TARGET_LOC, pickle_protocol=4)
