import os
import torch
import numpy as np
from pprint import pprint
from datetime import datetime

DATA_PATH = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000'
TARGET_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000/all'
DATA_TARGET_LOC = os.path.join(TARGET_LOC, 'data.pt')
LABELS_TARGET_LOC = os.path.join(TARGET_LOC, 'labels.pt')

def get_files_path(data_path): 
    data_path_list = []
    labels_path_list = []
    
    for cur_dir, _, files in os.walk(data_path):
        if cur_dir.endswith('all'):
           for file in files:
               path = os.path.join(cur_dir,file)
               name_of_file = path.rsplit('/',1)[-1] 
               
               if name_of_file.startswith('data'): 
                   data_path_list.append(path)
               if name_of_file.startswith('labels'): 
                   labels_path_list.append(path)
                   
    data_path_list = sorted(data_path_list, key=lambda path: path.rsplit('/',1)[-2])
    labels_path_list = sorted(labels_path_list, key=lambda path: path.rsplit('/',1)[-2])
    pprint(data_path_list)
    pprint(labels_path_list)
    return data_path_list, labels_path_list

def compose_data(paths, is_data):
    all_data = None
    for path in paths:
        print('Processing file: ', path)
        tic = datetime.now()
        if is_data:
           #import pdb;pdb.set_trace();
           loaded_numpy = np.expand_dims(np.load(path), axis=0)
        else:
           loaded_numpy = np.expand_dims(np.load(path), axis=0)
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
#os.makedirs(TARGET_LOC, exist_ok=True)
os.makedirs(TARGET_LOC)
torch.save(data, DATA_TARGET_LOC, pickle_protocol=4)
torch.save(lables, LABELS_TARGET_LOC, pickle_protocol=4)
