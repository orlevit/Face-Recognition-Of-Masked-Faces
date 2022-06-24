# Test if the cosine similaruty in "find_threshold_10_kfold_like_old_cosine_sim.py" is correct. so loaded the LFw benchmark with the insightface methods and then check the cosine similarity with my code. The results was the same for the LFW as the test of the Insightface code.
# I keep it if a case in the futher may comes up.
import os
import sys
import pickle
import math
import torch
import sklearn
import random
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
from time import time
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.realpath('../model_training'))
from config import SPLIT_TRAIN, TRAIN_DS_IND, VALID_DS_IND, WHOLE_DATA_BATCH, TEST_DS_IND, THRESHOLDS_INTERVALS
from models_architecture import NeuralNetwork5
from helper import get_optimizer, select_train_valid 

#BASE_MODELS_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'
BASE_MODELS_PATH = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning/r100-arcface-org_masked/model'
#TRAIN_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/train/combinedV1_multi_masks_350000/all'
#TRAIN_DATA_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'data.pt')
#TRAIN_LABELS_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'labels.pt')

# Combined version2
#TRAIN_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/prepare_run/bin/bins_files/train/all'
#TRAIN_DATA_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'data.npy')
#TRAIN_LABELS_LOC = os.path.join(TRAIN_BASE_DATA_LOC, 'labels.npy')
#TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV2'
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '300000_pairs_same_masks_hidden4096_NeuralNetwork5_lr1e-07_32_D30_05_2022_T16_54_35_404599.pt')

# Data
# Combined version1
#TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV1'
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000_pairs_batch_all_hidden4096_NeuralNetwork5_lr1e-05_32_D20_02_2022_T18_53_58_770221.pt')

# Combined version2
TEST_BASE_DATA_LOC = '/RG/rg-tal/orlev/project/insightface/datasets/nomask/lfw.bin'

### Retrain
# Version1
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-05_32_D17_06_2022_T16_43_43_708447.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-06_32_D17_06_2022_T16_41_52_082341.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV1_NeuralNetwork5_lastHidden4096_lr1e-07_32_D17_06_2022_T16_40_49_981712.pt')
# Version2
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-05_32_D17_06_2022_T16_44_18_742560.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-06_32_D17_06_2022_T16_42_13_910480.pt')
#MODEL_PATH = os.path.join(BASE_MODELS_PATH, '350000pairsV2_NeuralNetwork5_lastHidden4096_lr1e-07_32_D17_06_2022_T16_41_08_309996.pt')

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class LFold:
    def __init__(self, n_splits=2, shuffle=False, random_state=42):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


class EmbbedingsDataset(Dataset):
    def __init__(self, data, labels, ds_ind, mask=None):
        if ds_ind == TRAIN_DS_IND:
            self.data = data[:, np.repeat(mask, 2), :]
            self.labels = labels[0, :len(mask)][mask]
        elif ds_ind == VALID_DS_IND:
            self.data = data[:, ~np.repeat(mask, 2), :]
            self.labels = labels[0, :len(mask)][~mask]
        else:
            self.data = data
            self.labels = labels
        print(f'Indication; {ds_ind}, Total pairs: {len(self.labels)}, number of pairs same: {sum(self.labels)}, number of pairs diff: {len(self.labels) - sum(self.labels)}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embeddings1 = np.concatenate(np.expand_dims([emb[2 * idx, :] for emb in self.data],axis=0), axis=1)
        embeddings2 = np.concatenate(np.expand_dims([emb[2 * idx + 1, :] for emb in self.data],axis=0), axis=1)
        label = self.labels[idx]
        return embeddings1, embeddings2, label

def create_dataloaders_valid(train_data_loc, train_labels_loc, split_train, valid_ds_ind, batch_size):
    train_data = torch.load(train_data_loc)
    train_labels = torch.load(train_labels_loc)
    print(f'Train data shape: {train_data.shape}, train labels shape: {train_labels.shape}')

    # Numbers for split the data properly 
    mask =select_train_valid(train_data, split_train)

    validDataset = EmbbedingsDataset(train_data, train_labels, valid_ds_ind, mask)

    if batch_size == WHOLE_DATA_BATCH:
       batch_size_valid = len(validDataset)
    else:
       batch_size_valid = batch_size

    valid_dataloader = DataLoader(validDataset, batch_size=batch_size_valid, shuffle=False)

    return valid_dataloader

def creae_dataloaders_test(test_data_loc, test_labels_loc, split_train, batch_size, test_ds_ind):
    test_data = torch.load(test_data_loc)
    test_labels = torch.load(test_labels_loc)
    print(f'Test data shape: {test_data.shape}, test labels shape: {test_labels,shape}')
  
    testDataset = EmbbedingsDataset(test_data, test_labels, test_ds_ind)

    if batch_size == WHOLE_DATA_BATCH:
       batch_size_test = len(testDataset)
    else:
       batch_size_test = batch_size

    test_dataloader = DataLoader(testDataset, batch_size=batch_size_test, shuffle=False)

    return test_dataloader 

def one_epoch_run_threshold(train_dataloader, model, device, threshold, train_ind=False):
    last_loss = 0.
    running_loss = 0.
    running_classificatin_loss = 0.
    tic = datetime.now()

    for i, data in enumerate(train_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 

        max_threshold, max_accuracy = find_a_threshold(emb1, emb2, labels, train_ind, best_threshold=threshold)
        running_classificatin_loss += max_accuracy
        
    run_time = round((datetime.now() - tic).total_seconds(), 1)
    avg_classificatin_loss = running_classificatin_loss / len(train_dataloader)

    return 0, avg_classificatin_loss, run_time

def find_best_threshold(valid_dataloader, model, device, fold_num):
    for i, data in enumerate(valid_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
        max_threshold, max_accuracy = find_a_threshold(emb1, emb2, labels, train_ind=True, best_threshold=None)
        print(f" THe best accuracy on the {fold_num}/(10) fold train set is:{max_accuracy} and the threshold is:{max_threshold}")
    return max_threshold

def get_similarity(emb1, emb2):
    emb1_0 = emb1[:, 0, :]
    emb2_0 = emb2[:, 0, :]
   # concat_dim1 = np.concatenate([emb1_0, emb1_1], axis = -1)
   # concat_dim2 = np.concatenate([emb2_0, emb2_1], axis = -1)
    cos_sim = np.sum(np.square(np.subtract(emb1_0, emb2_0)), 1)
    return cos_sim


def find_a_threshold(emb1, emb2, labels, train_ind, best_threshold):
    thresholds = np.arange(0, 4, 0.01)
    nrof_thresholds = len(thresholds)
    accuracies = np.zeros(nrof_thresholds)
    emb1 = emb1.cpu().detach().numpy()
    emb2 = emb2.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    labels[labels == -1] = 0

    if train_ind:
       for threshold_idx, threshold in enumerate(thresholds):
           accuracies[threshold_idx] = calculate_accuracy(threshold, emb1, emb2, labels)

       max_threshold_idx = np.argmax(accuracies)
       max_threshold = thresholds[max_threshold_idx]
       max_accuracy = accuracies[max_threshold_idx]
    else:
       max_accuracy = calculate_accuracy(best_threshold, emb1, emb2, labels)
       max_threshold = best_threshold
    return max_threshold, max_accuracy

def calculate_accuracy(threshold, emb1, emb2, actual_issame):
    cos_sim = get_similarity(emb1, emb2)
    predict_issame = np.less(cos_sim, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    acc = float(tp + tn) / len(actual_issame)

    return  acc

def get_test_data_by_indices(data, labels, indices):
    mask = np.zeros(len(labels), dtype=bool)
    mask[indices] = True
    selected_labels = list(np.array(labels)[mask])
    print(data.shape,mask.shape, sum(selected_labels))
    selected_data = data[:,np.repeat(mask, 2),:]
    testDataset = EmbbedingsDataset(selected_data, selected_labels, TEST_DS_IND)
    test_dataloader = DataLoader(testDataset, batch_size=len(testDataset), shuffle=False)

    return test_dataloader



def load_bin(path, image_size=(112,112)):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  #py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  #py3
    data_list = []
    for flip in [0, 1]:
        data = nd.empty(
            (len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for i in range(len(issame_list) * 2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][i][:] = img
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return (data_list, issame_list)

def get_embeddings(mx_model,
                   batch_size = 32,
                   data_extra=None,
                   label_shape=None):
    print('testing verification..')
    data_list, issame_list = load_bin(TEST_BASE_DATA_LOC, (112,112))
    model = mx_model
    embeddings_list = []
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    time_consumed = 0.0
    if label_shape is None:
        _label = nd.ones((batch_size, ))
    else:
        _label = nd.ones(label_shape)
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            #print(_data.shape, _label.shape)
            time0 = datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data, ), label=(_label, ))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label, ))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    #embeddings1 = embeddings[0::2]
    #embeddings2 = embeddings[1::2]
    #embs = np.stack([embeddings1, embeddings2], axis = 0)
    embs = embeddings[np.newaxis,:,:]
    print(embs.shape)
    print('infer time', time_consumed)
    return embs, issame_list

def get_model(prefix):
    image_size = [112, 112]
    print('image_size', image_size)
    ctx = mx.gpu(0)
    nets = []
    prefix = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning/r100-arcface-org_masked/model' 
    vec = (prefix,0)
    epochs = []
    pdir = os.path.dirname(prefix)
    for fname in os.listdir(pdir):
        if not fname.endswith('.params'):
            continue
        _file = os.path.join(pdir, fname)
        if _file.startswith(prefix):
            epoch = int(fname.split('.')[0].split('-')[1])
            epochs.append(epoch)
    epochs = sorted(epochs, reverse=True)

    print('model number', len(epochs))
    time0 = datetime.now()
    for epoch in epochs:
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        #arg_params, aux_params = ch_dev(arg_params, aux_params, ctx)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
        model.bind(data_shapes=[('data', (32, 3, image_size[0],
                                          image_size[1]))])
        model.set_params(arg_params, aux_params)
        nets.append(model)
    time_now = datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())
    return model

def main():
    model = get_model(BASE_MODELS_PATH)
    k_fold = LFold(n_splits=10, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #valid_dataloader = create_dataloaders_valid(TRAIN_DATA_LOC, TRAIN_LABELS_LOC, SPLIT_TRAIN, VALID_DS_IND, WHOLE_DATA_BATCH) 
    #threshold = find_best_threshold(valid_dataloader, model, device)

    #test_data = np.load(os.path.join(cur_dir, sub_dir, 'data.npy'))
    #test_labels = list(np.load(os.path.join(cur_dir, sub_dir, 'labels.npy'))[0,:])
    test_data, test_labels = get_embeddings(model, batch_size = 32)
    print(f'data shape: {test_data.shape}, labels length: {len(test_labels)}')
    indices = np.arange(len(test_labels))
    classification_list = []
    thresholds_list = []

    blockPrint()
    for fold_num,(train_indices, test_indices) in enumerate(k_fold.split(indices)):
        train_dataloader = get_test_data_by_indices(test_data, test_labels, train_indices)
        threshold = find_best_threshold(train_dataloader, model, device, fold_num)
        test_dataloader = get_test_data_by_indices(test_data, test_labels, test_indices)
        print(f'Split number:{fold_num}\nTrain data set size(90%): {len(train_dataloader.dataset)} Test data set size(10%): {len(test_dataloader.dataset)}')
        avg_loss, avg_classificatin_loss, run_time = one_epoch_run_threshold(test_dataloader, model, device, threshold, train_ind=False)
        classification_list.append(avg_classificatin_loss)
        print(f'Test: run_time:{run_time}, loss={avg_loss}, classification accuracy:{avg_classificatin_loss}, thresold:{threshold}')
        thresholds_list.append(threshold)
    enablePrint()
    print(f'{np.mean(classification_list)}+-{np.std(classification_list)}, avg_thresold={np.mean(thresholds_list)} ,{thresholds_list=}')
    print('---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()

