# Make sure that SPLIT_DATA_SAVE is divieded without leftovers, otherwise the while loop will truncate the last elements
# only one bin file - chunckisize the file for small pieces by SPLIT_DATA_SAVE
# virtual environment tf_gpu_py36
#import h5py
import os
import gc
import json
import torch
import pickle
import shutil
import argparse
from mxnet import nd
import mxnet as mx
import numpy as np
from pprint import pprint
from datetime import datetime
from itertools import groupby

# Constants
MAX_NUMER_OF_DATA_FOR_MODEL = 700000
SPLIT_DATA_SAVE = 10000
BATCH_SIZE = 1
NUMBER_OF_MODELS = 7
IMAGE_SIZE = [112, 112]
MASK_NAME = 'org' # if change to "no" keep the model as orgmodel!
#BIN_LOC_SKELETON = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/a{}mask/a{}mask.bin'
DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/images_masked_crop/cfp'
#BIN_LOC_SKELETON = '/RG/rg-tal/orlev/datasets/original_ds/MFR2_bg/a{}mask/a{}mask.bin'
BIN_LOC_SKELETON = os.path.join(DATA_LOC, 'a{}mask/a{}mask_c.bin')
#BIN_LOC = '/RG/rg-tal/orlev/project/insightface/datasets/nomask/agedb30.bin'
#DBS_MODELS_DATA_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data'
MODEL_DIR_LOC_SKELETON = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning/r100-arcface-{}_masked'

BIN_LOC = ""
MODEL_DIR_LOC = ""

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
        #print(f'embeddings::{ii}, bb:{bb}')
        bb = min(ba + batch_size, data.shape[0])
        count = bb - ba
        _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
        if data_extra is None:
            db = mx.io.DataBatch(data=(_data, ), label=(_label, ))
        else:
            db = mx.io.DataBatch(data=(_data, _data_extra), label=(_label, ))
        import pdb;pdb.set_trace();
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        #importpdb; pdb.set_trace();
        _embeddings = net_out[0]#.asnumpy()
        #_embeddings = net_out[0].asnumpy()
        #print(_embeddings.shape)
        if embeddings is None:
            embeddings = nd.zeros((data.shape[0], _embeddings.shape[1]))
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



# this is read every file of  data and than save(instaed read a all sata and thaen save)
def forward_bins_through_models(image_size, batch_size, args):
    bins_dir_loc = BIN_LOC_SKELETON.format(args.model, args.model)
    model_dir_loc = MODEL_DIR_LOC_SKELETON.format(args.model)
    models_dir, epochs = set_models_epochs(model_dir_loc)
    models, models_thresholds, models_names = set_models(models_dir, epochs, batch_size, image_size)
    #models, models_thresholds, models_names = [models[args.model]], [models_thresholds[args.model]], [models_names[args.model]]
    data_dict, all_bins_names_dict = bin_loc_to_data(bins_dir_loc, image_size)
    for dstype_name, (data_lists, issame_lists) in data_dict.items():
        for data_i, (data_list, issame_list) in enumerate(zip(data_lists, issame_lists)):
            bin_data_length = data_list[0].shape[0]
            issame_length = len(issame_list)

            for model_i, model in enumerate(models):
                aleardy_processed = 0; split_num = 0; 
                while aleardy_processed <= bin_data_length: #and aleardy_processed < 700000:#aleardy_processed < 200000:# 500000:#MAX_NUMER_OF_DATA_FOR_MODEL:
                      #if aleardy_processed >= 100000:#200000:
                      #if aleardy_processed >= 350000:
                      #if aleardy_processed == 440000:
                         coverted_data = None
                         time1 = datetime.now()
                         bin_data_org = data_list[0][aleardy_processed : min(aleardy_processed + SPLIT_DATA_SAVE, bin_data_length)]
                         bin_data_flip = data_list[1][aleardy_processed : min(aleardy_processed + SPLIT_DATA_SAVE, bin_data_length)]
                         embedding_org = create_embeddings(bin_data_org, model, batch_size,  None, None)
                         embedding_flip = create_embeddings(bin_data_flip, model, batch_size, None, None)
                         embeddings_joined = embedding_org + embedding_flip
                         if coverted_data is None:
                            coverted_data = embeddings_joined
                            coverted_issame = nd.array(issame_list[aleardy_processed : min(aleardy_processed + SPLIT_DATA_SAVE, bin_data_length)])
    
                         time2 = datetime.now()
                         diff = time2 - time1
                         time1 = datetime.now()
                         #curr_model_name = models_dir[model_i].rsplit('/',1)[-1]
                         curr_model_name = models_names[0]
                         print(f'Emb created. db:{dstype_name}({data_i+1}/{len(data_lists)}). model:{curr_model_name}({model_i+1}/{len(models)}). Processed:({aleardy_processed}/{bin_data_length}) . time(min):{diff.total_seconds()/60}')
    
                         dst_dir = os.path.join(DATA_LOC, dstype_name, 'db_' + all_bins_names_dict[dstype_name][data_i], models_names[0])
                         os.makedirs(dst_dir, exist_ok=True)
                         time2 = datetime.now()
                         diff = time2 - time1
                         print('MAKING THE DIRE IF NOT EXISTS', diff.total_seconds()/60)
                         time1 = datetime.now()
                         #import pdb; pdb.set_trace();
                         #filename = os.path.join(dst_dir, f'data_{aleardy_processed}.h5py')
                         #f1 = h5py.File(filename, "w")
                         #f1.create_dataset('dataset',data= coverted_data.asnumpy(), shape=(1000,512), dtype=np.float32)
                         #f1.close()
                         #np.save(os.path.join(dst_dir, f'data_{aleardy_processed}.npy'), coverted_data)
                         mx.nd.save(os.path.join(dst_dir, f'data_{aleardy_processed}.npy'), coverted_data) # This is essental 11
                         time2 = datetime.now()
                         diff = time2 - time1
                         print('data saving time', diff.total_seconds()/60)
                         time1 = datetime.now()
                         if split_num <= issame_length:
                            mx.nd.save(os.path.join(dst_dir, f'labels_{aleardy_processed}.npy'), coverted_issame)
                         time2 = datetime.now()
                         diff = time2 - time1
                         print('Save file of labels to locaton time(min)', diff.total_seconds()/60)
                         time1 = datetime.now()
                         del coverted_data
                         del coverted_issame
                         del embedding_org
                         del embedding_flip
                         del embeddings_joined
                         time2 = datetime.now()
                         diff = time2 - time1
                         print('Delete time(min)', diff.total_seconds()/60)
                         time1 = datetime.now()
                         gc.collect()
                         time2 = datetime.now()
                         diff = time2 - time1
                         print('Collect time(min)', diff.total_seconds()/60)
                         aleardy_processed += SPLIT_DATA_SAVE
                         split_num += 1

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
        
        
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Which model to create.')

    return parser.parse_args()

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
import pdb; pdb.set_trace();
args = parse_arguments()
forward_bins_through_models(IMAGE_SIZE, BATCH_SIZE, args)
