import os
import numpy as np
import mxnet as mx
import pandas as pd
from mxnet import nd
from PIL import Image
from datetime import datetime
from ijbc_config import *


def change_file_name_img_frame(filename):
    dir_name, file_name_w_end = filename.split('/')
    file_name, ending = file_name_w_end.split('.')
    trailed_zeros_name = '%06d' % int(file_name)
    if dir_name == 'img':
        new_file_name = '1' + trailed_zeros_name + '.'+ending
    elif dir_name == 'frames':
        new_file_name = '2' + trailed_zeros_name + '.'+ending
    else:
        raise(f'The name not exists; {filename}')
    return new_file_name

def template_to_loc(tmp_id):
    def _template_to_loc(tmp_id):
        single_row = TEMPLATE_DF[TEMPLATE_DF['TEMPLATE_ID'] == tmp_id]
        filename = single_row['FILENAME'].values[0]
        sub_id = str(single_row['SUBJECT_ID'].values[0])
        new_filename = change_file_name_img_frame(filename)
        loc = os.path.join(IMAGES_DIR, sub_id, new_filename)
        return loc, sub_id

    img_loc, sub_id = _template_to_loc(tmp_id)
 
    return tmp_id, img_loc, sub_id

def set_models(models_dir, epochs, image_size, batch_size):
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
        models_names.append(model_loc.rsplit('/',1)[-1].split('-')[-1].split('_')[0])

        # load model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_loc + '/model', epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
        model.bind(data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        nets.append(model)

    time_now = datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    return nets, models_thresholds, models_names

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

def create_embeddings(data, model, batch_size):
    _label = nd.ones((batch_size, ))
    embeddings = None
    ba = 0
    ii=0; bb=0;
    while ba < data.shape[0]:
        bb = min(ba + batch_size, data.shape[0])
        count = bb - ba
        data = mx.nd.array(data) ############
        _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
        db = mx.io.DataBatch(data=(_data, ), label=(_label, ))
        model.forward(db, is_train=False)
        net_out = model.get_outputs()
        _embeddings = net_out[0].asnumpy()
        if embeddings is None:
            embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
        embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
        ba = bb
        ii+=1

    return embeddings


def forward_data_through_models(models, images_data, image_size, batch_size):
    all_data = None
    for model_i, model in enumerate(models):
        embedding_org = create_embeddings(images_data[0], model, batch_size)
        embedding_flip = create_embeddings(images_data[1], model, batch_size)
        embeddings_joined = embedding_org + embedding_flip # Normalized?!?!
        embeddings_joined = embeddings_joined[np.newaxis, :]
        if all_data is None:
           all_data = embeddings_joined
        else:
           all_data = np.vstack((all_data,  embeddings_joined))

    return all_data

    

def read_and_prepare_img(img_loc):
    read_img = np.array(Image.open(img_loc))
    read_img_flip = read_img.copy()
    img = np.transpose(read_img, (2,0,1))[np.newaxis, :] 
    img_flip = np.transpose(np.flip(read_img_flip, axis=2),(2,0,1))[np.newaxis, :]
    
    return (img, img_flip)

def add_embs_to_arr(ie_idx, tmp_id, ie_arr, img_embs):
    if ie_arr is None:
       ie_arr = img_embs
       ie_idx = [tmp_id]
    else:
       ie_arr = np.hstack((ie_arr, img_embs))
       ie_idx.append(tmp_id)
    return ie_arr, ie_idx

def templeate_to_embs(tmp_id,  models, ie_idx, ie_arr):
    if tmp_id in ie_idx:
       single_row = TEMPLATE_DF[TEMPLATE_DF['TEMPLATE_ID'] == tmp_id]
       sub_id = str(single_row['SUBJECT_ID'].values[0])
       index = ie_idx.index(tmp_id)
       return ie_arr[:, index, :], sub_id, ie_arr, ie_idx
    else: 
       print(f'Tmemplate ID is not exists in the pre created list: {tmp_id}')
       tmp_id, img_embs, sub_id = create_templeate_to_embs(tmp_id,  models)
       ie_arr, ie_idx = add_embs_to_arr(ie_idx, tmp_id, ie_arr, img_embs)
       return img_embs, sub_id, ie_arr, ie_idx

def create_templeate_to_embs(tmp_id, models):
    tmp_id, img_loc, sub_id = template_to_loc(tmp_id)
    images = read_and_prepare_img(img_loc)
    img_embs = forward_data_through_models(models, images, IMAGE_SIZE, batch_size=1)

    return tmp_id, img_embs, sub_id

def ijbc_data_and_labels(indices, transfer_models, ie_idx, ie_arr):
    batch_size = len(indices) 
    # the retrun data and labels or the composed model
#    if test_index == TEST_TYPE_ORG:
#       data = np.zeros((7, 2 * batch_size, 512))
#    else:
#       data = np.zeros((1, 2 * batch_size, 512))

    data = np.zeros((len(transfer_models), 2 * batch_size, 512))
    labels = np.zeros((batch_size))

    for ii, one_index in enumerate(indices):
        single_row = MATCH_DF.iloc[one_index]
        tmp_id1 = single_row['ENROLL_TEMPLATE_ID']
        tmp_id2 = single_row['VERIF_TEMPLATE_ID']

        img1_embs, sub_id1, ie_arr, ie_idx = templeate_to_embs(tmp_id1, transfer_models, ie_idx, ie_arr)
        img2_embs, sub_id2, ie_arr, ie_idx = templeate_to_embs(tmp_id2, transfer_models, ie_idx, ie_arr)

        data[:, ii, :] = np.squeeze(img1_embs)
        data[:, ii + 1, :] = np.squeeze(img2_embs)
         
        if sub_id1 == sub_id2:
           labels[ii] = 1

    return data, labels, ie_idx, ie_arr
