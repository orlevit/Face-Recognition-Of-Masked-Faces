# based on the implementation of "one_bin_to_data.py'.
# joined all the .npy arrays in directory for that one db (e.g.covid19) will joined all the models inside it's directory to one, in order to test it performence.
# will create a joint db to all the directries in the INPUT folder. will be like "db_test_models" but fot the datasets
# only one bin file
# virtual environment tf_gpu_py36
import os
import gc
import json
import torch
import pickle
import shutil
from mxnet import nd
import mxnet as mx
import numpy as np
from pprint import pprint
from datetime import datetime
from itertools import groupby

# Constants
NUMBER_OF_MODELS = 7
#DBS_MODELS_DATA_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/images_masked_crop/sample_ROF/wsnp/covid19/combined'
DBS_MODELS_DATA_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/tmp'
MODEL_DIR_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning'

# ------------   Also run this when change train to test ---------------
INPUT_MODELS_DATA_LOC = os.path.join(DBS_MODELS_DATA_LOC, 'test')
TARGET_MODELS_DATA_LOC = os.path.join(DBS_MODELS_DATA_LOC, 'db_for_test_combinedV1')
# ---------------------------------------------------------------------
# The change in this functon relative to the dame fucntion in the joined_test_dbs.py file is changing 2 to 3 in the location with the arrow: "<---"
def get_files_loc_models_data(data_loc):
    data_files_loc = []
    label_files_loc = []

    for curr_dir1, subFolders, _ in os.walk(data_loc):
        for sub_folder in sorted(subFolders, key=lambda x: x.rsplit('/',1)[-1]):
            for curr_dir2, subFolders2, _ in os.walk(os.path.join(curr_dir1, sub_folder)):
                for sub_folder2 in sorted(subFolders2, key=lambda x: x.rsplit('/',1)[-1]):
                    for curr_dir3, _, files in os.walk(os.path.join(curr_dir2, sub_folder2)):
                        for file_name in files:
                            path = os.path.join(curr_dir3,file_name)
                            name_of_file = path.rsplit('/',1)[-1] 
                            if file_name.endswith('data.npy'):
                                data_files_loc.append(os.path.join(curr_dir3,file_name))

                            if file_name.endswith('labels.npy'):
                                label_files_loc.append(os.path.join(curr_dir3,file_name))

    joined_loc = [(i,j) for i,j in zip(data_files_loc, label_files_loc)]
    sorted_joined_loc = sorted(joined_loc,key=lambda x: (x[0].split('/')[-4],x[0].split('/')[-3],x[0].split('/')[-2])) #<--- (-2)->(-3) and 9-3)->(-2)
    gs_joined_loc = [list(v) for i, v in groupby(sorted_joined_loc, lambda x: x[0].split('/')[-3])] #<---
    
    return gs_joined_loc


def joined_models(gs_joined_loc, target_data_loc):
    for dbs_locs in gs_joined_loc:
        combined_dbs_for_model = None
        db_name = dbs_locs[0][0].split('/')[-3]
        pprint(f'---------------  + db name:{db_name} + ---------------')
        for db_i, masked_db in  enumerate(dbs_locs, 1):
            tic = datetime.now()
    
            loaded_db = np.expand_dims(np.load(masked_db[0]), axis=0)
            loaded_lbl = np.expand_dims(np.load(masked_db[1]), axis=0)
    
            if combined_dbs_for_model is None:
                combined_dbs_for_model = loaded_db
                combined_lbls_for_model = loaded_lbl
            else: 
                #import pdb;pdb.set_trace();
                combined_dbs_for_model = np.vstack((combined_dbs_for_model, loaded_db))
                combined_lbls_for_model = np.vstack((combined_lbls_for_model, loaded_lbl))
            toc = datetime.now() 
            model_name = masked_db[0].split('/')[-2]
            pprint(f'Model name: {model_name}; time: {toc-tic}')
        pprint(f'Loaded: {db_i} DBl: {db_name}')
        save_loc = os.path.join(target_data_loc, db_name)
        os.makedirs(save_loc, exist_ok=True)
        np.save(save_loc + '/data.npy', combined_dbs_for_model)
        np.save(save_loc + '/labels.npy', combined_lbls_for_model)
        
        

def get_files_loc_models(data_loc):
    data_files_loc = []
    label_files_loc = []

    for curr_dir, subFolder, files in os.walk(DATA_LOC):
        for file_name in files:
            if file_name.endswith('data.npy'):
                data_files_loc.append(os.path.join(curr_dir,file_name))

            if file_name.endswith('labels.npy'):
                label_files_loc.append(os.path.join(curr_dir,file_name))

    joined_loc = [(i,j) for i,j in zip(data_files_loc, label_files_loc)]
    sorted_joined_loc = sorted(joined_loc,key=lambda x: (x[0].split('/')[-2].split('_')[0]))
    
    return sorted_joined_loc

def combine_datasets(dbs_loc, dst_loc):
    comb_data = None
    for i, loc in enumerate(dbs_loc):
        curr_data = np.expand_dims(np.load(loc[0]), axis=0)
        pprint(f'Loaded: {loc}')
        if comb_data is None:
            comb_data = curr_data
        else:
            comb_data = np.concatenate((comb_data, curr_data), axis=0)
    labels_file = np.load(loc[1])
    torch.save(labels_file, os.path.join(dst_loc, 'all_labels.pt'), pickle_protocol=4)
    torch.save(comb_data, os.path.join(dst_loc, 'all_data.pt'), pickle_protocol=4)
    
gs_joined_loc = get_files_loc_models_data(INPUT_MODELS_DATA_LOC)
import pdb;pdb.set_trace();
joined_models(gs_joined_loc, TARGET_MODELS_DATA_LOC)

