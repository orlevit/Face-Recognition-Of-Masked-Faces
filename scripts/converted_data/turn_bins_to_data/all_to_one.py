# joined all the "all files" (got form "chuncks_all.py"). will jouned 7 files along the first axis
import os
import torch
import numpy as np
from glob import glob
from pprint import pprint
from datetime import datetime

# version 1
DATA_PATH = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV1'
TARGET_LOC = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV1/all/20k_pairs'
# verson 2
#DATA_PATH = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV2'
#TARGET_LOC = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/combinedV2/all/20k_pairs'
DATA_TARGET_LOC = os.path.join(TARGET_LOC, 'data.pt')
LABELS_TARGET_LOC = os.path.join(TARGET_LOC, 'labels.pt')

def get_files_path(data_path): 
    data_path_list = []
    labels_path_list = []
    
    #for cur_dir, _, files in os.walk(data_path):
    cur_dir = 'sf'
    for file in glob(data_path + '*/**/train/all_350k_pairs/*.npy'):
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
           loaded_numpy = torch.load(path) # load mxnet/torch or numpy
        else:
           loaded_numpy = torch.load(path) # load mxnet/torch or numpy
        if all_data is None:
            all_data = loaded_numpy
        else:
            all_data = np.concatenate((all_data, loaded_numpy), axis=0) # to contatenate according o an axis - change to poper use
            
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
