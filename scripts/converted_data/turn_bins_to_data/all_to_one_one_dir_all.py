import os
import torch
import mxnet as mx
import numpy as np
from glob import glob
from pprint import pprint
from datetime import datetime

DATA_PATH = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/a*mask/train/combined/*.npy'
TARGET_LOC = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/all'
DATA_TARGET_LOC = os.path.join(TARGET_LOC, 'data.npy')
LABELS_TARGET_LOC = os.path.join(TARGET_LOC, 'labels.npy')
def get_files_path(data_path): 
    print(data_path)
    data_path_list = []
    labels_path_list = []
    
    #for cur_dir2, _, files in os.walk(data_path):
    for file in glob(data_path):
        #path = os.path.join(cur_dir2,file)
        path = file
        name_of_file = path.rsplit('/',1)[-1] 
        if name_of_file.startswith('data'): 
            data_path_list.append(path)
        if name_of_file.startswith('labels'): 
            labels_path_list.append(path)
        
    data_path_list = sorted(data_path_list, key=lambda path: path.rsplit('/')[-4])
    labels_path_list = sorted(labels_path_list, key=lambda path: path.rsplit('/')[-4])
    pprint(data_path_list)
    pprint(labels_path_list)
    #import pdb;pdb.set_trace();
    return data_path_list, labels_path_list

def compose_data(paths, is_data):
    all_data = None
    for path in paths:
        print('Processing file: ', path)
        tic = datetime.now()
        loaded_numpy = np.load(path, allow_pickle=True)
        #loaded_numpy = loaded_numpy
        #import pdb;pdb.set_trace();
        loaded_numpy = np.expand_dims(loaded_numpy, axis=0)
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
