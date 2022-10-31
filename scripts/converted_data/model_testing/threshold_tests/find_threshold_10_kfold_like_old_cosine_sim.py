# Checking if adding all the vectors of the 7 models (512*7) and then test the cosine similarity  between the the concatenated vectors of 2 images if they are the same. recreate the checking is similar to theInsight face vectors similarity check. The results is 50% and the threshold id 0
# No model is necessry to be uploaded it is pnly the embeddings.
import os
import sys
import math
import torch
import random
import numpy as np
from time import time
from datetime  import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.realpath('../../model_training'))
from config import SPLIT_TRAIN, TRAIN_DS_IND, VALID_DS_IND, WHOLE_DATA_BATCH, TEST_DS_IND, THRESHOLDS_INTERVALS
from models_architecture import *
from helper import EmbbedingsDataset, get_optimizer, select_train_valid 


# Data
# Combined version1
#TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV1'

# Combined version2
TEST_BASE_DATA_LOC = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/lfw_test/db_for_test_combinedV2'

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

def one_epoch_run_threshold(train_dataloader, device, threshold, train_ind=False):
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

def find_best_threshold(valid_dataloader, device, fold_num):
    for i, data in enumerate(valid_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
        max_threshold, max_accuracy = find_a_threshold(emb1, emb2, labels, train_ind=True, best_threshold=None)
        print(f" THe best accuracy on the {fold_num}/(10) fold train set is:{max_accuracy} and the threshold is:{max_threshold}")
    return max_threshold

def get_similarity(emb1, emb2):
    emb1_0 = emb1[:, 0, :]
    emb1_1 = emb1[:, 1, :]
    emb1_2 = emb1[:, 2, :]
    emb1_3 = emb1[:, 3, :]
    emb1_4 = emb1[:, 4, :]
    emb1_5 = emb1[:, 5, :]
    emb1_6 = emb1[:, 6, :]
    emb2_0 = emb2[:, 0, :]
    emb2_1 = emb2[:, 1, :]
    emb2_2 = emb2[:, 2, :]
    emb2_3 = emb2[:, 3, :]
    emb2_4 = emb2[:, 4, :]
    emb2_5 = emb2[:, 5, :]
    emb2_6 = emb2[:, 6, :]

    concat_dim1 = np.concatenate([emb1_0, emb1_1, emb1_2, emb1_3, emb1_4, emb1_5, emb1_6], axis = -1)
    concat_dim2 = np.concatenate([emb2_0, emb2_1, emb2_2, emb2_3, emb2_4, emb2_5, emb2_6], axis = -1)
    cos_sim = np.sum(np.square(np.subtract(concat_dim1, concat_dim2)), 1)
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
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k_fold = LFold(n_splits=10, shuffle=False)

    #valid_dataloader = create_dataloaders_valid(TRAIN_DATA_LOC, TRAIN_LABELS_LOC, SPLIT_TRAIN, VALID_DS_IND, WHOLE_DATA_BATCH) 
    #threshold = find_best_threshold(valid_dataloader, model, device)

    for cur_dir, subFolders, _ in os.walk(TEST_BASE_DATA_LOC):
        for sub_dir in subFolders:
            test_data = np.load(os.path.join(cur_dir, sub_dir, 'data.npy'))
            test_labels = list(np.load(os.path.join(cur_dir, sub_dir, 'labels.npy'))[0,:])
            print(f'Data loc:{os.path.join(cur_dir, sub_dir)}, data shape: {test_data.shape}, labels length: {len(test_labels)}')
            indices = np.arange(len(test_labels))
            classification_list = []
            thresholds_list = []

            blockPrint()
            for fold_num,(train_indices, test_indices) in enumerate(k_fold.split(indices)):
                train_dataloader = get_test_data_by_indices(test_data, test_labels, train_indices)
                threshold = find_best_threshold(train_dataloader, device, fold_num)
                test_dataloader = get_test_data_by_indices(test_data, test_labels, test_indices)
                print(f'Split number:{fold_num}\nTrain data set size(90%): {len(train_dataloader.dataset)} Test data set size(10%): {len(test_dataloader.dataset)}')
                avg_loss, avg_classificatin_loss, run_time = one_epoch_run_threshold(test_dataloader,  device, threshold, train_ind=False)
                classification_list.append(avg_classificatin_loss)
                print(f'Test: run_time:{run_time}, loss={avg_loss}, classification accuracy:{avg_classificatin_loss}, thresold:{threshold}')
                thresholds_list.append(threshold)
            enablePrint()
            print(f'Images:{sub_dir}: {np.mean(classification_list)}+-{np.std(classification_list)}, avg_thresold={np.mean(thresholds_list)} ,{thresholds_list=}')
            print('---------------------------------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
