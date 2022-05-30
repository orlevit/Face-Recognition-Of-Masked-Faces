import os
import mxnet as mx
import torch
import numpy as np
from pprint import pprint
from datetime import datetime
LIMIT_TO_READ = 300000
MASK = 'no'
DATA_PATH = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/a{MASK}mask/train/db_a{MASK}mask_a{MASK}mask/org_model'
TARGET_LOC = f'/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/a{MASK}mask/train/combined'
DATA_TARGET_LOC = os.path.join(TARGET_LOC, 'data.npy')
LABELS_TARGET_LOC = os.path.join(TARGET_LOC, 'labels.npy')

def get_files_path(data_path): 
    print(data_path)
    data_path_list = []
    labels_path_list = []
    
    for cur_dir2, _, files in os.walk(data_path):
       for file in files:
           path = os.path.join(cur_dir2,file)
           name_of_file = path.rsplit('/',1)[-1] 
           if int(name_of_file.split('_')[-1].split('.')[0]) < LIMIT_TO_READ:
                if name_of_file.startswith('data'): 
                    data_path_list.append(path)
                if name_of_file.startswith('labels'): 
                    labels_path_list.append(path)
          
    data_path_list = sorted(data_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    labels_path_list = sorted(labels_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    pprint(data_path_list)
    pprint(labels_path_list)
    #import pdb;pdb.set_trace();
    return data_path_list, labels_path_list

def compose_data(paths, is_data):
    all_data = None
    for path in paths:
        print('Processing file: ', path)
        tic = datetime.now()
        loaded = mx.nd.load(path)
        loaded_numpy = loaded[0].asnumpy()
        #import pdb;pdb.set_trace();
        #loaded_numpy = loaded_numpy
        #loaded_numpy = np.expand_dims(loaded_numpy, axis=0)
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
np.save(DATA_TARGET_LOC, data)
np.save(LABELS_TARGET_LOC, np.asarray(lables))
