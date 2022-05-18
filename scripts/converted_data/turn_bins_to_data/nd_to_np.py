import re
import os
import numpy as np
import mxnet as mx
from pprint import pprint
from datetime import datetime

BASE_PATH = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000'
TARGET_LOC = os.path.join(BASE_PATH, '{}_all')
#DATA_PATH = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000/sunglasses_model'
#TARGET_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/db_bin_multi_masks_350000/sunglasses_model_all'
#DATA_TARGET_LOC = os.path.join(TARGET_LOC, 'data.npy')
#LABELS_TARGET_LOC = os.path.join(TARGET_LOC, 'labels.npy')

def get_files_path(data_path): 
    data_path_list = []
    labels_path_list = []
    
    for cur_dir, _, files in os.walk(data_path):
        for file in files:
            path = os.path.join(cur_dir,file)
            name_of_file = path.rsplit('/',1)[-1] 
            #if int(re.split('_|\.', name_of_file)[-2]) < 200000:  # This is because the script is runnin while the foles are created
            if name_of_file.startswith('data_'): 
                data_path_list.append(path)
            if name_of_file.startswith('labels_'): 
                labels_path_list.append(path)
                
    data_path_list = sorted(data_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    labels_path_list = sorted(labels_path_list, key=lambda path: int(path.rsplit('/',1)[-1].split('.')[0].split('_')[-1]))
    #pprint( data_path_list)
    #pprint( labels_path_list)
    return data_path_list, labels_path_list

def compose_data(paths):
    all_data = None
    for path in paths:
        print('Processing file: ', path)
        tic = datetime.now()
        loaded = mx.nd.load(path)
        loaded_numpy = loaded[0].asnumpy()
        del loaded
        
        if all_data is None:
            all_data = loaded_numpy
        else:
            all_data = np.concatenate((all_data, loaded_numpy), axis=0)
            
        toc = datetime.now()
        print('Time: ',toc-tic)
    return all_data 


for root,  dirs, files in os.walk(BASE_PATH):
    for name in dirs:
        data_path_list, labels_path_list = get_files_path(os.path.join(root, name))
        data = compose_data(data_path_list)
        lables = compose_data(labels_path_list)
        #os.makedirs(TARGET_LOC, exist_ok=True)
        target_location = TARGET_LOC.format(name)
        os.makedirs(target_location)
        data_target_loc = os.path.join(target_location, 'data.npy')
        labels_target_loc = os.path.join(target_location, 'labels.npy')
        np.save(data_target_loc, data)
        np.save(labels_target_loc, lables)
