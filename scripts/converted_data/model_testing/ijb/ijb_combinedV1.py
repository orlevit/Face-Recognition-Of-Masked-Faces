import os
import gc
import sys
import argparse
import pickle
import timeit
import warnings
from pathlib import Path
from  datetime import datetime
import cv2
import matplotlib
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
import sklearn
from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from mxnet.gluon.data import Dataset, DataLoader
from prettytable import PrettyTable
from skimage import transform as trans
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

sys.path.append('/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_training')
from models_architecture import *

matplotlib.use('Agg')
warnings.filterwarnings("ignore")
BASE_LOADED_EMBS_DIR = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/converted_data/model_testing/ijb'
MODEL_DIR_LOC = '/RG/rg-tal/orlev/Face-Recognition-Of-Masked-Faces/scripts/rec_run/models/transfer_learning'
BASE_COMPOSED_MODELS_PATH = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/models'
MODEL_PATH = os.path.join(BASE_COMPOSED_MODELS_PATH, '40000pairsV1_reduce8_seed42_NeuralNetwork15_lastHidden4096_lr1e-05_32_D05_07_2022_T17_49_25_335043.pt')

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--image-path', default='', type=str, help='')
parser.add_argument('--result-dir', default='.', type=str, help='')
parser.add_argument('--gpu', default='0', type=str, help='gpu id')
parser.add_argument('--batch-size', default=128, type=int, help='')
parser.add_argument('--job', default='insightface', type=str, help='job name')
parser.add_argument('-es', '--emb-size', type=int, help='embedding size')
parser.add_argument('-lif', '--load-img-feats', type=int, help='Load images features')
parser.add_argument('--target',
                    default='IJBC',
                    type=str,
                    help='target, set to IJBC or IJBB')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

target = args.target
image_path = args.image_path
META_PATH = os.path.join('%s/meta' % image_path, '%s_meta' % target.upper())
LOOSE_PATH = os.path.join(image_path, 'ijb', target.upper())
result_dir = args.result_dir
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = args.job
batch_size = args.batch_size
load_img_feats = args.load_img_feats

def load_composed_model():
    print(MODEL_PATH)
    model = NeuralNetwork15()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH)
    model_state_dict = checkpoint["model_state_dict"]
    best_threshold = checkpoint["threshold"]
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model

class DatasetIJB(Dataset):
    def __init__(self, root, lines, align=True):
        self.src = np.array(
            [[30.2946, 51.6963], [65.5318, 51.5014], [48.0252, 71.7366],
             [33.5493, 92.3655], [62.7299, 92.2041]],
            dtype=np.float32)
        self.src[:, 0] += 8.0
        self.lines = lines
        self.img_root = root
        self.align = align

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        each_line = self.lines[idx]
        name_lmk_score = each_line.strip().split(' ')  # "name lmk score"
        img_name = os.path.join(self.img_root, name_lmk_score[0])
        img = cv2.imread(img_name)

        if self.align:
            landmark = np.array([float(x) for x in name_lmk_score[1:-1]],
                                dtype=np.float32)
            landmark = landmark.reshape((5, 2))
            #
            assert landmark.shape[0] == 68 or landmark.shape[0] == 5
            assert landmark.shape[1] == 2
            if landmark.shape[0] == 68:
                landmark5 = np.zeros((5, 2), dtype=np.float32)
                landmark5[0] = (landmark[36] + landmark[39]) / 2
                landmark5[1] = (landmark[42] + landmark[45]) / 2
                landmark5[2] = landmark[30]
                landmark5[3] = landmark[48]
                landmark5[4] = landmark[54]
            else:
                landmark5 = landmark
            #
            tform = trans.SimilarityTransform()
            tform.estimate(landmark5, self.src)
            #
            M = tform.params[0:2, :]
            img = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, 112, 112), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return mx.nd.array(input_blob)


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
        model = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=None)
        model.bind(data_shapes=[('data', (2 * batch_size, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        nets.append(model)
    time_now = datetime.now()
    diff = time_now - time0
    print('model loading time', diff.total_seconds())

    return nets, models_thresholds, models_names


def foreward_embs(dataset, batch_size, size, use_flip_test):
    def combine_models_embs(all_embs, emb):
        emb_expended = emb[np.newaxis, :, :]
        if all_embs is None:
           all_embs = emb_expended
        else: 
           all_embs = np.concatenate((all_embs, emb_expended), axis=0)
        return all_embs

    all_embs = None
    models_dir, epochs = set_models_epochs(MODEL_DIR_LOC)
    for model_i, (model_dir, epoch) in enumerate(zip(models_dir, epochs)):
        img_feats = extract_parallel(model_dir, epoch, dataset, batch_size, size)
        all_embs = combine_models_embs(all_embs, img_feats)
        model_name =  model_dir.rsplit('/', 1)[-1]
        mx.gpu(0).empty_cache()
        print(f'Model model: {model_name}, embeddings shape:{all_embs.shape}')

    if use_flip_test:
        all_embs = all_embs[:, :, 0:all_embs.shape[-1] //
                                    2] + all_embs[:, :, all_embs.shape[-1] // 2:]
    else:
        all_embs = all_embs[:, 0:all_embs.shape[1] // 2]

    return all_embs


def extract_parallel(prefix, epoch, dataset, batch_size, size):
    # init
    model_list = list()
    num_ctx = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    num_iter = 0
    feat_mat = mx.nd.zeros(shape=(len(dataset), 2 * size))

    def batchify_fn(data):
        return mx.nd.concat(*data, dim=0)

    data_loader = DataLoader(dataset,
                             batch_size,
                             last_batch='keep',
                             num_workers=8,
                             thread_pool=True,
                             prefetch=16,
                             batchify_fn=batchify_fn)
    symbol, arg_params, aux_params = mx.module.module.load_checkpoint(
        prefix + '/model', epoch)
    all_layers = symbol.get_internals()
    symbol = all_layers['fc1_output']

    # init model list
    for i in range(num_ctx):
        model = mx.mod.Module(symbol, context=mx.gpu(i), label_names=None)
        model.bind(for_training=False,
                   data_shapes=[('data', (2 * batch_size, 3, 112, 112))])
        model.set_params(arg_params, aux_params)
        model_list.append(model)

    # extract parallel and async
    num_model = len(model_list)
    #import pdb;pdb.set_trace();
    for image in tqdm(data_loader):
        data_batch = mx.io.DataBatch(data=(image, ))
        model_list[num_iter % num_model].forward(data_batch, is_train=False)
        feat = model_list[num_iter %
                          num_model].get_outputs(merge_multi_context=True)[0]
        feat = mx.nd.L2Normalization(feat)
        feat = mx.nd.reshape(feat, (-1, size * 2))
        feat_mat[batch_size * num_iter:batch_size * num_iter +
                 feat.shape[0], :] = feat.as_in_context(mx.cpu())
        num_iter += 1

    del model
    gc.collect()
    return feat_mat.asnumpy()

def read_template_media_list(path):
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


def read_template_pair_list(path):
    pairs = pd.read_csv(path, sep=' ', header=None).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label

def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((img_feats.shape[0], len(unique_templates), img_feats.shape[2]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t, ) = np.where(templates == uqt)
        #face_norm_feats = img_feats[ind_t]
        face_norm_feats = img_feats[:, ind_t, :]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            #ind_m = np.array([1,2]);ct=2; #############3
            if ct == 1:
                media_norm_feats += [face_norm_feats[:, ind_m, :]]
            else:  # image features from the same video will be aggregated into one feature
                mnf_one = np.stack([np.mean(face_norm_feats[model_i, ind_m, :], axis=0, keepdims=True) for model_i in range(0, face_norm_feats.shape[0])], axis=0)
                media_norm_feats += [mnf_one]

        media_norm_feats = np.stack(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[:, count_template, : ] = np.squeeze(np.sum(media_norm_feats, axis=0))
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats_list = [sklearn.preprocessing.normalize(template_feats_model) for template_feats_model in template_feats]
    template_norm_feats = np.stack(template_norm_feats_list)
    return template_norm_feats, unique_templates


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    composed_model = load_composed_model()
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for c, sub in enumerate(sublists):
        feat1 = torch.tensor(template_norm_feats[:, np.squeeze(template2id[p1[sub]]), :])
        feat2 = torch.tensor(template_norm_feats[:, np.squeeze(template2id[p2[sub]]), :])
        feat1 = torch.swapaxes(feat1, 0, 1)
        feat2 = torch.swapaxes(feat2, 0, 1)
        outputs = composed_model(feat1.to(device).float(), feat2.to(device).float())
        #similarity_score = np.sum(feat1 * feat2, -1)
        #score[s] = similarity_score.flatten()
        score[sub] = outputs.detach().cpu().numpy().flatten()
        torch.cuda.empty_cache()
        del feat1, feat2, outputs
        gc.collect()

        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    
    import pdb;pdb.set_trace();
    checkpoint = torch.load(MODEL_PATH)
    model_state_dict = checkpoint["model_state_dict"]
    best_threshold = checkpoint["threshold"]
    return score



# # Step1: Load Meta Data

assert target == 'IJBC' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join(META_PATH, '%s_face_tid_mid.txt' % target.lower()))
#templates, medias = read_template_media_list(
#    os.path.join('%s/meta' % image_path,
#                 '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join(META_PATH, '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# =============================================================
# load image features
# format:
#img_path = '%s/loose_crop' % image_path
# =============================================================
start = timeit.default_timer()
#img_path = '%s/loose_crop' % image_path
#img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_path = os.path.join(LOOSE_PATH, 'loose_crop')
#img_list_path = os.path.join(META_PATH, '%s_name_5pts_score.txt' % target.lower())
img_list_path = os.path.join(META_PATH, '%s_name_5pts_score.txt' % target.lower())

img_list = open(img_list_path)
files = img_list.readlines()
dataset = DatasetIJB(root=img_path, lines=files, align=True)

if args.load_img_feats:
  img_feats = np.load(os.path.join(BASE_LOADED_EMBS_DIR, 'img_feats', target.lower(), 'img_feats.npy'))
else:
   img_feats = foreward_embs(dataset, args.batch_size, size=args.emb_size, use_flip_test=use_flip_test)

faceness_scores = []
for each_line in files:
    name_lmk_score = each_line.split()
    faceness_scores.append(name_lmk_score[-1])

faceness_scores = np.array(faceness_scores).astype(np.float32)

stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))

# # Step3: Get Template Features

# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）


if use_norm_score:
    img_input_feats = img_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats**2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    img_input_feats = img_input_feats * faceness_scores[np.newaxis, :, np.newaxis]
else:
    img_input_feats = img_input_feats

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores

# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

save_path = result_dir + '/%s_result' % target

if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % job)
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
# x_labels = [1/(10**x) for x in np.linspace(6, 0, 6)]
x_labels = [10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        # tpr_fpr_row.append('%.4f' % tpr[min_index])
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10**-6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
# plt.show()
fig.savefig(os.path.join(save_path, '%s.pdf' % job))
print(tpr_fpr_table)

