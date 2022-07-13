# this is evaluation method only for the original model - with 0..4 threhold

import mxnet as mx
import numpy as np
from datetime  import datetime

def join_ouputs(all_outputs, outputs):
    if all_outputs is None:
       return outputs
    else:
       return torch.cat((all_outputs, outputs), dim=0)

def calculate_accuracy(threshold, outputs, actual_issame):
    predict_issame = np.greater(outputs, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    acc = float(tp + tn) / len(outputs)

    return  acc

def find_a_threshold(all_emb1, all_emb2, labels, train_ind, best_threshold):
    thresholds = np.arange(0, 4, 0.01)
    nrof_thresholds = len(thresholds)
    accuracies = np.zeros(nrof_thresholds)
    all_emb1 = all_emb1.cpu().detach().numpy()
    all_emb2 = all_emb2.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    diff = np.squeeze(np.subtract(all_emb1, all_emb2))
    outputs = np.sum(np.square(diff), 1)

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


def one_epoch_run_original(train_dataloader, model, device, train_ind, best_threshold=None):
    all_emb1 = None
    all_emb2 = None
    all_labels = None
    tic = datetime.now()

    for i, data in enumerate(train_dataloader):
        emb1, emb2, labels = data
        emb1, emb2, labels = emb1.to(device), emb2.to(device), labels.to(device) 
        all_emb1 = join_ouputs(all_emb1, emb1)
        all_emb2 = join_ouputs(all_emb2, emb2)
        all_labels = join_ouputs(all_labels, labels)

 
    threshold, max_accuracy = find_a_threshold(all_emb1, all_emb2, all_labels, train_ind, best_threshold)
    run_time = round((datetime.now() - tic).total_seconds(), 1)

    return -1, max_accuracy, threshold, run_time

