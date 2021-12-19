"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
import cv2
import math
import datetime
import pickle
from sklearn.decomposition import PCA
import mxnet as mx
from mxnet import ndarray as nd
import sklearn.metrics as metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc_tpr_fpr(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca = 0):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    
    tprs = np.zeros((nrof_folds))
    fprs = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    #print('pca', pca)
    
    if pca==0:
      diff = np.subtract(embeddings1, embeddings2)
      dist = np.sum(np.square(diff),1)
    
    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], _ = calculate_accuracy(threshold, dist, actual_issame)

    tpr = tprs
    fpr = fprs
          
    return tpr, fpr

def calculate_roc_tpr_fpr2(thresholds, scores, actual_issame, nrof_folds=10):
    nrof_pairs = min(len(actual_issame), scores.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)
    indices = np.arange(nrof_pairs)
    tpr = np.zeros((nrof_thresholds))
    fpr = np.zeros((nrof_thresholds))

    for threshold_idx, threshold in enumerate(thresholds):
        tpr[threshold_idx], fpr[threshold_idx], _ = calculate_accuracy2(threshold, scores, actual_issame)

    return tpr, fpr

def calculate_roc2(thresholds, learned_thresholds, scores, actual_issame, nrof_folds=10):
    nrof_pairs = min(len(actual_issame), scores.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        _, _, accuracy[fold_idx] = calculate_accuracy2(learned_thresholds[test_set], scores[test_set], actual_issame[test_set])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy2(threshold, scores[test_set], actual_issame[test_set])
    return tprs, fprs, accuracy

def calculate_accuracy2(thresholds, scores, actual_issame):
    #import pdb; pdb.set_trace()
    predict_issame = np.less(scores, thresholds)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / scores.size
    return tpr, fpr, acc

def calculate_roc(thresholds,
                  learned_threshold,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        _, _, accuracy[fold_idx] = calculate_accuracy(learned_threshold, dist[test_set], actual_issame[test_set])
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])

    return tprs, fprs, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    #print(true_accept, false_accept)
    #print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate2(learned_thresholds, scores, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(-1, 1, 0.005)
    tpr, fpr, accuracy = calculate_roc2(thresholds, learned_thresholds, scores, np.asarray(actual_issame), nrof_folds=nrof_folds)
    tpr_graph, fpr_graph = calculate_roc_tpr_fpr2(thresholds, scores, np.asarray(actual_issame), nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, tpr_graph, fpr_graph 


def evaluate(learned_threshold, embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       learned_threshold,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    tpr_graph, fpr_graph = calculate_roc_tpr_fpr(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)

    return tpr, fpr, accuracy, tpr_graph, fpr_graph 


def load_bin(path, image_size):
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

def create_embeddings(data, model, batch_size, data_extra, label_shape):
    if label_shape is None:
        _label = nd.ones((batch_size, ))
    else:
        _label = nd.ones(label_shape)
    if data_extra is not None:
        _data_extra = nd.array(data_extra)
    embeddings = None
    ba = 0
    ii=0
    while ba < data.shape[0]:
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

def score_calc(raw_embeddings_list, raw_thresholds):
    import pdb; pdb.set_trace()
    # calulate cosine similarity
    cos_sim_list = []
    for model_emb in raw_embeddings_list:
        embeddings1 = model_emb[0::2]
        embeddings2 = model_emb[1::2]
        _, norm1 = sklearn.preprocessing.normalize(embeddings1,return_norm=True)
        _, norm2 = sklearn.preprocessing.normalize(embeddings2,return_norm=True)
        cos_sim = np.sum(embeddings1 * embeddings2, axis=1) / (norm1 * norm2) 
        cos_sim_list.append(cos_sim)
     
    # calculate weighted average 
    cos_sim_arr = np.asarray(cos_sim_list)
    beta = 10
    weights = np.exp(beta * cos_sim_arr) / np.sum(np.exp(beta * cos_sim_arr), axis=0)

    fused_score = np.sum(cos_sim_arr * weights, axis=0)
    raw_thresholds_arr = np.expand_dims(np.asarray(raw_thresholds, dtype=np.float), axis=1)
    threhold_cos = (2- raw_threholds_arr) / 2
    threhold_cos_weighted = np.sum(threhold_cos * weights, axis=0)
     
    return np.arccos(fused_score), np.arccos(threhold_cos_weighted)

def test(data_set, mx_models, batch_size, raw_thresholds, nfolds=10, data_extra=None, label_shape=None, ROC = False, target_name=None, roc_dst=None):
    print('testing verification..')
    #print(len(data_set[0]),,data_set[1].shape)
    data_list = data_set[0]
    issame_list = data_set[1]
    model = mx_models
    raw_embeddings_list = []
    time_consumed = 0.0
    time0 = datetime.datetime.now()
    
    #raw_embeddings_list = np.load('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/test/raw_arr.npy') 
    for j in range(len(mx_models)):
        print(j, mx_models[j])
        embedding_org = create_embeddings(data_list[0], model[j], batch_size, data_extra, label_shape)
        embedding_flip = create_embeddings(data_list[1], model[j], batch_size, data_extra, label_shape)
       
        embeddings_joined = embedding_org + embedding_flip
        raw_embeddings_list.append(embeddings_joined)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    time_consumed += diff.total_seconds()
    print('calculate the embedding creation time(minutes):', time_consumed/60)
    # convert to angels(from euclidian)
    scores, learned_thresholds = score_calc(raw_embeddings_list, raw_thresholds)

    #_xnorm = 0.0
    #_xnorm_cnt = 0
    #for embed in embeddings_list:
    #    for i in range(embed.shape[0]):
    #        _em = embed[i]
    #        _norm = np.linalg.norm(_em)
    #        #print(_em.shape, _norm)
    #        _xnorm += _norm
    #        _xnorm_cnt += 1
    #_xnorm /= _xnorm_cnt

    _xnorm = -999
    acc1 = 0.0
    std1 = 0.0
    #tpr, fpr, accuracy, tpr_graph, fpr_graph = evaluate(threshold, embeddings, issame_list, nrof_folds=nfolds) 
    tpr, fpr, accuracy, tpr_graph, fpr_graph = evaluate2(learned_thresholds, scores, issame_list, nrof_folds=nfolds) 
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    roc_auc = []
    for single_tpr, single_fpr in zip(tpr, fpr):
       roc_auc.append(metrics.auc(single_fpr, single_tpr))

    # Save image as figure
    if ROC:
       roc_auc_graph = metrics.auc(fpr_graph, tpr_graph)
       fig = plt.figure()
       plt.title('Receiver Operating Characteristic')
       plt.plot(fpr_graph, tpr_graph, 'b', label = 'AUC = %0.2f' % roc_auc_graph)
       plt.legend(loc = 'lower right')
       plt.plot([0, 1], [0, 1],'r--')
       plt.xlim([0, 1])
       plt.ylim([0, 1])
       plt.ylabel('True Positive Rate')
       plt.xlabel('False Positive Rate')
       path = os.getcwd()
       plt.savefig(os.path.join(roc_dst, 'images', str(target_name)+'_ROC.jpg'))
       with open(os.path.join(roc_dst, 'arrays', str(target_name) +'.pkl'), 'wb') as f:
           pickle.dump({'fpr':fpr_graph, 'tpr':tpr_graph, 'auc':roc_auc_graph}, f)
        
    return acc1, std1, acc2, std2, _xnorm, roc_auc


def set_models_epochs(models_loc):
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
                models_dir.append(curr_dir)# + '/model')

    else:
        epochs = [int(x) for x in vec[1].split('|')]
        models_dir = prefix

    return models_dir, epochs

def set_models(models_dir, epochs, batch_size):
    models_thresholds = []
    models_thresh_names = []
    time0 = datetime.datetime.now()
    for model_loc, epoch in zip(models_dir, epochs):
        print('loading', model_loc, epoch)
        # add threshold
        with open(model_loc + '/model_threshold.txt' ,"r") as threshold_file:
             text = threshold_file.read()
        models_thresholds.append(float(text.rsplit('_', 1)[0].rsplit('_',1)[-1]))
        models_thresh_names.append(model_loc.rsplit('/',1)[-1])

        # load model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_loc + '/model', epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        nets.append(model)

    time_now = datetime.datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    return nets, models_thresholds, models_thresh_names

def set_target(args):
    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
        path = os.path.join(args.data_dir, name + ".bin")
        if os.path.exists(path):
            print('loading.. ', name)
            data_set = load_bin(path, image_size)
            ver_list.append(data_set)
            ver_name_list.append(name)

    return ver_list, ver_name_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--thresholds', type=str, help='The treshold selected')
    group.add_argument('--threshold-dir', default='', type=str, help='The tresholds file location, calc the avg threshold of the selected epoch ')
    # general
    parser.add_argument('--data-dir', default='', help='')
    parser.add_argument('--model',
                        default='../model/softmax,50',
                        help='path to load model.')
    parser.add_argument('--target',
                        default='lfw,cfp_ff,cfp_fp,agedb_30',
                        help='test targets.')
    parser.add_argument('--plot-roc', default=True, help='true or false to plot ROC curve')
    parser.add_argument('--roc-name', help='test targets.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--batch-size', default=32, type=int, help='')
    parser.add_argument('--max', default='', type=str, help='')
    parser.add_argument('--mode', default=0, type=int, help='')
    parser.add_argument('--nfolds', default=10, type=int, help='')
    args = parser.parse_args()
    #sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
    #import face_image
    #prop = face_image.load_property(args.data_dir)
    #image_size = prop.image_size

    image_size = [112, 112]
    print('image_size', image_size)
    ctx = mx.gpu(args.gpu)
    nets = []
    models_dir, epochs = set_models_epochs(args.model)
    print('model:', models_dir, 'epochs', epochs) 

    models, models_thresholds, models_thresh_names = set_models(models_dir, epochs, args.batch_size)
    ver_list, ver_name_list = set_target(args)

    roc_dst, roc_name = args.roc_name.rsplit('/', 1)

    if not os.path.exists(os.path.join(roc_dst, 'images')):
       os.makedirs(os.path.join(roc_dst, 'images'))
    if not os.path.exists(os.path.join(roc_dst, 'arrays')):
       os.makedirs(os.path.join(roc_dst, 'arrays'))

    for i in range(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, roc_auc = test(ver_list[i], models, args.batch_size, models_thresholds, args.nfolds, ROC=args.plot_roc, target_name=roc_name, roc_dst=roc_dst)
 
        roc_auc_mean, roc_auc_std = np.mean(roc_auc), np.std(roc_auc)
        print('The thresholds are:', [f'{i}={j}' for i,j in zip(models_thresh_names, models_thresholds)])
        print('[%s]XNorm: %f' % (ver_name_list[i], xnorm))
        print('[%s]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], acc1, std1))
        print('%1.5f+-%1.5f' %(acc2, std2))
        print('%1.5f+-%1.5f' %(roc_auc_mean, roc_auc_std))
