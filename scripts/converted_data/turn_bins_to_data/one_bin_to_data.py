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
BATCH_SIZE = 1
NUMBER_OF_MODELS = 7
IMAGE_SIZE = [112, 112]
#BIN_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/multi_masks_350000.bin'
BIN_LOC = '/home/orlev/work/project/insightface/datasets/eyemask/lfw.bin'
DBS_MODELS_DATA_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/tmp'
MODEL_DIR_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning'

# ------------   Also run this when change train to test ---------------
INPUT_MODELS_DATA_LOC = os.path.join(DBS_MODELS_DATA_LOC, 'test')
TARGET_MODELS_DATA_LOC = os.path.join(DBS_MODELS_DATA_LOC, 'db_test_models')
INPUT_MODELS_LOC = os.path.join(DBS_MODELS_DATA_LOC, 'db_test_models')
TARGET_MODELS_LOC = os.path.join(DBS_MODELS_DATA_LOC, 'all_test')
# ---------------------------------------------------------------------

def set_models_epochs(models_loc):
    print(f'Modls: {models_loc}')
    vec = models_loc.split(',')
    prefix = models_loc.split(',')[0]
    epochs = []
    models_dir = []
    if len(vec) == 1: # In my case: t is a directiry of multiply models
       for curr_dir, subFolder, files in os.walk(vec[0]):
           for file_name in files:
            if file_name.endswith('.params'):
                epoch = int(file_name.split('.')[0].split('-')[1])
                epochs.append(epoch)
                models_dir.append(curr_dir)

    else:
        epochs = [int(x) for x in vec[1].split('|')]
        models_dir = prefix

    s_models_dir = sorted(models_dir,key=lambda x: (x.rsplit('/',1)[-1]))
    s_models_ind = [s_models_dir.index(ii) for ii in models_dir]
    s_epochs = [x for _, x in sorted(zip(s_models_ind, epochs))]

    return s_models_dir, s_epochs

def set_models(models_dir, epochs, batch_size, image_size):
    nets = []
    models_names = []
    models_thresholds = []
    time0 = datetime.now()
    for model_loc, epoch in zip(models_dir, epochs):
        print('loading', model_loc, epoch)
        # add threshold
        with open(model_loc + '/model_threshold.txt' ,"r") as threshold_file:
             text = threshold_file.read()
        models_thresholds.append(float(text.rsplit('_', 1)[0].rsplit('_',1)[-1]))
        models_names.append(model_loc.rsplit('/',1)[-1].split('-')[-1].split('_')[0] + '_model')

        # load model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_loc + '/model', epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        model.bind(data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        nets.append(model)
        #return nets, models_thresholds, models_names                     ########################### for more than one models delete it

    time_now = datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    return nets, models_thresholds, models_names

def create_embeddings(data, model, batch_size, data_extra, label_shape):
    if label_shape is None:
        _label = nd.ones((batch_size, ))
    else:
        _label = nd.ones(label_shape)
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    embeddings = None
    ba = 0
    ii=0; bb=0;
    while ba < data.shape[0]:
        print(f'embeddings::{ii}, bb:{bb}')
        bb = min(ba + batch_size, data.shape[0])
        count = bb - ba
        _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
        if data_extra is None:
            db = mx.io.DataBatch(data=(_data, ), label=(_label, ))
        else:
            db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label, ))
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        _embeddings = net_out[0].asnumpy()
        #print(_embeddings.shape)
        if embeddings is None:
            embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
        embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
        ba = bb
        ii+=1

    return embeddings

def set_bins_loc(bins_loc):
    print(bins_loc)
    bins_files_loc = []
    for curr_dir, subFolder, files in os.walk(bins_loc):
        for file_name in files:
         if file_name.endswith('.bin'): #file_name.endswith('agedb30.bin') or file_name.endswith('lfw.bin'):    ##################################### changr to include additonal files
             bins_files_loc.append(os.path.join(curr_dir,file_name))
    sorted_bin_files_loc = sorted(bins_files_loc,key=lambda x: (x.rsplit('/',1)[-1]))
    gs_bin_files_loc = [list(v) for i, v in groupby(sorted_bin_files_loc, lambda x: x.rsplit('/',1)[-1])]
    test_list = gs_bin_files_loc[0]
    train_list = list(np.concatenate(gs_bin_files_loc[1:]))
    #bins_loc_dict = {single_list[0].rsplit('/',1)[-1].split('.')[0]:single_list for single_list in gs_bin_files_loc}
    return test_list, train_list

def load_bins(bins_files_loc, image_size):
    all_data_list = []
    all_issame_list = []
    # Get all the bins filesin a location
    print('f{bins_files_loc}:bins_files_loc')
    for bin_files_loc in bins_files_loc:
        print(bin_files_loc)
        try:
            with open(bin_files_loc, 'rb') as f:
                bins, issame_list = pickle.load(f)  #py2
        except UnicodeDecodeError as e:
            with open(bin_files_loc, 'rb') as f:
                bins, issame_list = pickle.load(f, encoding='bytes')  #py3

        data_list = []
        for flip in [0, 1]:
            data = mx.nd.empty(
                (len(issame_list) * 2, 3, image_size[0], image_size[1]))
            data_list.append(data)
        for i in range(len(issame_list) * 2):
            _bin = bins[i]
            img = mx.image.imdecode(_bin)
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = mx.nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][i][:] = img
            if i % 1000 == 0:
                print('loading bin', i)
        print(data_list[0].shape)
        all_data_list.append(data_list)
        all_issame_list.append(issame_list)
    return all_data_list, all_issame_list

def bin_loc_to_data(bins_dir_loc, image_size):
    bins_train_list = [bins_dir_loc]
    train_data_list, train_issame_list = load_bins(bins_train_list, image_size)
    data_dict = {'train': (train_data_list, train_issame_list)}
    bins_train_names = [bin_loc.rsplit('/')[-2] +'_'+ bin_loc.rsplit('/',1)[-1].split('.')[0] for bin_loc in bins_train_list]
    all_bins_names_dict = {'train': bins_train_names}
    return data_dict, all_bins_names_dict



# This is read every file of  data and than save(instaed read a all sata and thaen save)
def forward_bins_through_models(model_dir_loc, bins_dir_loc, image_size, batch_size):
    models_dir, epochs = set_models_epochs(MODEL_DIR_LOC)
    models, models_thresholds, models_names = set_models(models_dir, epochs, batch_size, image_size)
    data_dict, all_bins_names_dict = bin_loc_to_data(bins_dir_loc, image_size)
    
    import pdb; pdb.set_trace();
    
    for dstype_name, (data_lists, issame_lists) in data_dict.items():
        for data_i, (data_list, issame_list) in enumerate(zip(data_lists, issame_lists)):
            for model_i, model in enumerate(models):
                coverted_data = None
                time1 = datetime.now()
                embedding_org = create_embeddings(data_list[0], model, batch_size,  None, None)
                embedding_flip = create_embeddings(data_list[1], model, batch_size, None, None)
                embeddings_joined = embedding_org + embedding_flip
                if coverted_data is None:
                   coverted_data = embeddings_joined
                   coverted_issame = np.asarray(issame_list)
    
                time2 = datetime.now()
                diff = time2 - time1
                curr_model_name = models_dir[model_i].rsplit('/',1)[-1]
                print(f'Emb created. db:{dstype_name}({data_i+1}/{len(data_lists)}). model:{curr_model_name}({model_i+1}/{len(models)}). time(min):{np.round(diff.total_seconds()/60,2)}')
    
               # import pdb; pdb.set_trace()
                dst_dir = os.path.join(DATA_LOC, dstype_name, 'db_' + all_bins_names_dict[dstype_name][data_i], models_names[model_i])
                os.makedirs(dst_dir, exist_ok=True)
                np.save(os.path.join(dst_dir, 'data.npy'), coverted_data)
                np.save(os.path.join(dst_dir, 'labels.npy'), coverted_issame)
                del coverted_data
                del coverted_issame
                del embedding_org
                del embedding_flip
                del embeddings_joined
                gc.collect()

    print('Fiished: join_models_dbs funcion')

def get_files_loc_models_data(data_loc):
    data_files_loc = []
    label_files_loc = []

    for curr_dir, subFolder, files in os.walk(data_loc):
            if file_name.endswith('data.npy'):
                data_files_loc.append(os.path.join(curr_dir,file_name))

            if file_name.endswith('labels.npy'):
                label_files_loc.append(os.path.join(curr_dir,file_name))

    joined_loc = [(i,j) for i,j in zip(data_files_loc, label_files_loc)]
    sorted_joined_loc = sorted(joined_loc,key=lambda x: (x[0].split('/')[-4],x[0].split('/')[-2],x[0].split('/')[-3]))
    gs_joined_loc = [list(v) for i, v in groupby(sorted_joined_loc, lambda x: x[0].split('/')[-2])]
    
    return gs_joined_loc

def joined_models(gs_joined_loc, target_data_loc):
    for model_locs in gs_joined_loc:
        combined_dbs_for_model = None
        model_name = model_locs[0][0].split('/')[-2]
        pprint(f'---------------  + model name:{model_name} + ---------------')
        for db_i, masked_db in  enumerate(model_locs, 1):
            tic = datetime.now()
    
            loaded_db = np.load(masked_db[0])
            loaded_lbl = np.load(masked_db[1])
    
            if combined_dbs_for_model is None:
                combined_dbs_for_model = loaded_db
                combined_lbls_for_model = loaded_lbl
            else: 
                combined_dbs_for_model = np.vstack((combined_dbs_for_model, loaded_db))
                combined_lbls_for_model = np.hstack((combined_lbls_for_model, loaded_lbl))
            toc = datetime.now() 
            db_name = model_locs[0][0].split('/')[-3]
            pprint(f'DB name: {db_name}; time: {toc-tic}')
        pprint(f'Loaded: {db_i} DBs for model: {model_name}')
        save_loc = os.path.join(target_data_loc, model_name)
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
    
# Forward the bins files throught the different models and save them
forward_bins_through_models(MODEL_DIR_LOC, BIN_LOC, IMAGE_SIZE, BATCH_SIZE)

import pdb; pdb.set_trace();
gs_joined_loc = get_files_loc_models_data(INPUT_MODELS_DATA_LOC)
joined_models(gs_joined_loc, TARGET_MODELS_DATA_LOC)

# join the bins for each model together
dbs_loc = get_files_loc_models(INPUT_MODELS_LOC)
os.makedirs(TARGET_MODELS_LOC, exist_ok=True)
combine_datasets(dbs_loc, TARGET_MODELS_LOC)

# Remove all intermediate files
#shutil.rmtree(INPUT_MODELS_DATA_LOC)
#shutil.rmtree(TARGET_MODELS_DATA_LOC)
#shutil.rmtree(INPUT_MODELS_LOC)
