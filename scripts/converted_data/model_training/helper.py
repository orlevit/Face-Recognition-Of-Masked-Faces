import math
import torch
import random
import argparse
import itertools
import numpy as np
from torch import nn
from datetime import datetime
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader
from config import TRAIN_DS_IND, VALID_DS_IND, TEST_DS_IND, WHOLE_DATA_BATCH, THRESHOLDS_INTERVALS, LINEAR_INIT, BILINEAR_INIT, LOAD_MODEL, PRELOADED_MODEL_LOC, SEED, hsdict

#def select_train_valid(data, labels, ds_ind):
#    imgs_num, mask_imgs_num, total_mask_img_num = 154000,22000*0.8, 22000
#    if ds_ind == TRAIN_DS_IND:
#       selected_data = data[:, np.mod(np.arange(imgs_num), total_mask_img_num) < mask_imgs_num, :]
#       selected_labels = labels[np.mod(np.arange(imgs_num // 2), total_mask_img_num // 2) < mask_imgs_num // 2]
#    elif ds_ind == VALID_DS_IND:
#       selected_data = data[:, np.mod(np.arange(imgs_num), total_mask_img_num) >= mask_imgs_num, :]
#       selected_labels = labels[np.mod(np.arange(imgs_num // 2), total_mask_img_num // 2) >= mask_imgs_num // 2]
#    else:
#       selected_data = data
#       selected_labels = labels
#    return selected_data, selected_labels 

def select_train_valid(train_data, split_train):
    models_num, imgs_num, embs_num = train_data.shape
    total_pairs_mask_img_num = (imgs_num // 2) / models_num
    train_mask_imgs_num = math.ceil((total_pairs_mask_img_num) * split_train / 2.0) * 2
    valid_mask_imgs_num = int(total_pairs_mask_img_num - train_mask_imgs_num)
    mask_data_chunck = np.r_[np.ones(train_mask_imgs_num), np.zeros(valid_mask_imgs_num)].astype(bool)
    random.seed(SEED)
    random.shuffle(mask_data_chunck)
    mask = np.tile(mask_data_chunck, models_num)
    leftover = imgs_num//2 - len(mask)
    if leftover != 0:
       train_leftover = round(leftover * split_train)
       valid_leftover = leftover - train_leftover
       mask_additional = np.r_[np.ones(train_leftover), np.zeros(valid_leftover)].astype(bool)
       mask = np.concatenate((mask, mask_additional))
    return mask

#def select_train_valid(train_data, split_train, args):
#    models_num, imgs_num, embs_num = train_data.shape
#    train_mask_imgs_num = math.ceil((imgs_num // 8) * split_train / 2.0) * 2
#    valid_mask_imgs_num = int(imgs_num // 8 - train_mask_imgs_num)
#    mask_data_chunck = np.r_[np.ones(train_mask_imgs_num), np.zeros(valid_mask_imgs_num)].astype(bool)
#    random.seed(args.combination_number)
#    random.shuffle(mask_data_chunck)
#    mask = np.tile(mask_data_chunck, 4)
#    return mask

class EmbbedingsDataset(Dataset):
    def __init__(self, data, labels, ds_ind, mask=None):
        #self.data, self.labels = select_train_valid(data, labels, imgs_num, mask_imgs_num, total_mask_img_num, ds_ind)
#        self.data, self.labels = select_train_valid(data, labels, ds_ind)
        #mask_data = np.r_[np.ones(22000*5), np.zeros(int(22000 * 1.5)), np.ones(int(22000* 0.5))].astype(bool)
        #mask_labels = np.r_[np.ones(11000*5), np.zeros(int(11000 * 1.5)), np.ones(int(11000* 0.5))].astype(bool)
        #mask_data = np.r_[np.ones(int(22000*5.5)), np.zeros(int(22000* 1.5))].astype(bool)
        #mask_labels = np.r_[np.ones(int(11000*5.5)), np.zeros(int(11000* 1.5))].astype(bool)
        #if ds_ind == TRAIN_DS_IND:
        #    self.data = data[:, mask_data, :]
        #    self.labels = labels[mask_labels]
        #elif ds_ind == VALID_DS_IND:
        #    self.data = data[:, ~mask_data, :]
        #    self.labels = labels[~mask_labels]
        #else:
        #    self.data = data
        #    self.labels = labels
        #split_train_valid = 0.8
        #if ds_ind == TRAIN_DS_IND:
        #    self.data = data[:, :int(data.shape[1] * split_train_valid), :]
        #    self.labels = labels[:int(labels.shape[0] * split_train_valid)]
        #elif ds_ind == VALID_DS_IND:
        #    self.data = data[:, - int(data.shape[1] * split_train_valid):, :]
        #    self.labels = labels[- int(labels.shape[0] * split_train_valid):]
        #else:
        #    self.data = data
        #    self.labels = labels
#################
        #self.data, self.labels = select_train_valid(data, labels, ds_ind)
        ## Only Covid19 images
#        if ds_ind == TRAIN_DS_IND or ds_ind == VALID_DS_IND:
#            mask = mask[:11000]
#            data = data[:,:22000,:]
        if ds_ind == TRAIN_DS_IND:
            self.data = data[:, np.repeat(mask, 2), :]
            #self.labels = labels[:len(mask)][mask]
            self.labels = labels[0, :len(mask)][mask]
        elif ds_ind == VALID_DS_IND:
            self.data = data[:, ~np.repeat(mask, 2), :]
            #self.labels = labels[:len(mask)][~mask]
            self.labels = labels[0, :len(mask)][~mask]
        else:
            self.data = data
            self.labels = labels
        print(f'Indication; {ds_ind}, Total pairs: {len(self.labels)}, number of pairs same: {sum(self.labels)}, number of pairs diff: {len(self.labels) - sum(self.labels)}')
 #       self.labels = (self.labels[:,None] == np.arange(2)).astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embeddings1 = np.concatenate(np.expand_dims([emb[2 * idx, :] for emb in self.data],axis=0), axis=1)
        embeddings2 = np.concatenate(np.expand_dims([emb[2 * idx + 1, :] for emb in self.data],axis=0), axis=1)
        label = self.labels[idx]
        return embeddings1, embeddings2, label
   
def calculate_accuracy(threshold, outputs, actual_issame):
    predict_issame = np.greater(outputs, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    acc = float(tp + tn) / len(outputs)

    return  acc

def find_a_threshold(outputs, labels, train_ind, best_threshold):
    thresholds = np.arange(-1, 1, THRESHOLDS_INTERVALS)
    nrof_thresholds = len(thresholds)
    accuracies = np.zeros(nrof_thresholds)
    outputs = torch.squeeze(outputs)
    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    labels[labels == -1] = 0

    if train_ind:
       for threshold_idx, threshold in enumerate(thresholds):
           accuracies[threshold_idx] = calculate_accuracy(threshold, outputs, labels)

       max_threshold_idx = np.argmax(accuracies)
       max_threshold = thresholds[max_threshold_idx]
       max_accuracy = accuracies[max_threshold_idx]
    else:
       max_accuracy = calculate_accuracy(best_threshold, outputs, labels)
       max_threshold = best_threshold
    return max_threshold, max_accuracy

def join_ouputs(all_outputs, outputs):
    if all_outputs is None:
       return outputs
    else:
       return torch.cat((all_outputs, outputs), dim=0)

def add_l1(model, loss):
    l1_lambda = 0.001
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    final_loss = loss + l1_lambda * l1_norm
    return final_loss

def add_l2(model, loss):
    l2_lambda = 0.01
    l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
    final_loss = loss + l2_lambda * l2_norm
    return final_loss

def one_epoch_run(train_dataloader, optimizer, model, loss_fn, device, train_ind, best_threshold=None):
    last_loss = 0.
    running_loss = 0.
    running_classificatin_loss = 0.
    all_outputs = None
    all_labels = None
    tic = datetime.now()
    model.train(train_ind)
    for i, data in enumerate(train_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
        optimizer.zero_grad()
        outputs = model(emb1.float(), emb2.float())
        all_outputs = join_ouputs(all_outputs, outputs)
        all_labels = join_ouputs(all_labels, labels)
        #converted_labels = labels.type(torch.long)
        converted_labels = labels.type(torch.float)[:, None]
        if type(loss_fn).__name__ != 'BCELoss':
           converted_labels[converted_labels == 0] = -1
           converted_labels = converted_labels.type(torch.float)[:, None]

        loss = loss_fn(outputs, converted_labels.float())
        #loss = add_l2(model, loss)
        #loss = loss_fn(outputs, converted_labels)

        if train_ind:
           loss.backward()
           optimizer.step()

        running_loss += loss.item()
 
    threshold, max_accuracy = find_a_threshold(all_outputs, all_labels, train_ind, best_threshold)
    #if type(loss_fn).__name__ != 'CrossEntropyLoss':
    #   threshold, max_accuracy = find_a_threshold(all_outputs, all_labels, train_ind, best_threshold)
    #else:
    #   threshold = float('inf')
    #   import pdb;pdb.set_trace();
    #   max_accuracy = np.sum(all_outputs == all_labels) 
    run_time = round((datetime.now() - tic).total_seconds(), 1)
    avg_loss = running_loss / len(train_dataloader)

    return avg_loss, max_accuracy, threshold, run_time


def create_dataloaders(train_data_loc, train_labels_loc, test_data_loc, test_labels_loc, split_train, train_ds_ind, valid_ds_ind, batch_size, test_ds_ind):
    train_data = torch.load(train_data_loc)
    train_labels = torch.load(train_labels_loc)
    test_data = torch.load(test_data_loc)
    test_labels = torch.load(test_labels_loc)
  
    # Numbers for split the data properly 
    mask =select_train_valid(train_data, split_train)

    tic = datetime.now()
    trainDataset = EmbbedingsDataset(train_data, train_labels, train_ds_ind, mask)
    validDataset = EmbbedingsDataset(train_data, train_labels, valid_ds_ind, mask)
    testDataset = EmbbedingsDataset(test_data, test_labels, test_ds_ind)

    if batch_size == WHOLE_DATA_BATCH:
       batch_size_train = len(trainDataset) // 3 +1
       batch_size_valid = len(validDataset) // 3 +1
       batch_size_test = len(testDataset) // 3 +1
    else:
       batch_size_train = batch_size 
       batch_size_valid = batch_size
       batch_size_test = batch_size

    train_dataloader = DataLoader(trainDataset, batch_size=batch_size_train, shuffle=False)
    valid_dataloader = DataLoader(validDataset, batch_size=batch_size_valid, shuffle=False)
    test_dataloader = DataLoader(testDataset, batch_size=batch_size_test, shuffle=False)
    toc = datetime.now()
    print(f'Finish loading models: {toc-tic}')
    return train_dataloader, valid_dataloader, test_dataloader 

def get_optimizer(combination_number, model):
    all_combinations = list(itertools.product(*[v for k,v in hsdict.items()]))
    comb = all_combinations[combination_number - 1]
    #print(f'The selected hyperparameters: lr:{comb[0]}, beta1:{comb[1]}, beta2:{comb[2]}, weight_decay:{comb[3]}')
    print(f'The selected hyperparameters: lr:{comb[0]}')#, beta1:{comb[1]}, beta2:{comb[2]}, weight_decay:{comb[3]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=comb[0])#, betas=(comb[1], comb[2]), weight_decay=comb[3])
    
    return optimizer, comb

def parse_arguments():
    parser = argparse.ArgumentParser(description='do verification')
    parser.add_argument('-cn', '--combination-number', type=int, help='Number of the hyperparamet combinaion')
    args = parser.parse_args()

    return args

def initialize_weights(linear_const, bilinear_const):
    def inner(m):
        torch.manual_seed(SEED)
    
        if isinstance(m, nn.Linear):
            linear_bond = 1 / math.sqrt(linear_const)
            nn.init.uniform_(m.weight, -linear_bond, linear_bond)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -linear_bond, linear_bond)
                
        elif isinstance(m, nn.Bilinear):
            linear_bond = 1 / math.sqrt(bilinear_const)
            nn.init.uniform_(m.weight, -bilinear_const, bilinear_const)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -bilinear_const, bilinear_const)
    return inner

def save_model_state(epoch, model, optimizer, avg_loss, avg_ctrain_loss, threshold, ttime, avg_vloss, avg_cvalid_loss, vtime, model_path):
    param_dict = {
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'threshold': threshold,
                  'train':{
                            'time': ttime,
                            'avg_loss': avg_loss,
                            'avg_classificaton_loss': avg_ctrain_loss,
                          },
                  'valid':{
                            'time': vtime,
                            'avg_loss': avg_vloss,
                            'avg_classificaton_loss': avg_cvalid_loss,
                          }
                  }
    torch.save(param_dict, model_path)

def set_model(model, device):
    if LOAD_MODEL:
       model.load_state_dict(torch.load(PRELOADED_MODEL_LOC))
       print(f'Loaded previous model: {PRELOADED_MODEL_LOC}')
    else:
       model.apply(initialize_weights(LINEAR_INIT, BILINEAR_INIT))
       print('Initalized new model')
    model.to(device)
    return model
