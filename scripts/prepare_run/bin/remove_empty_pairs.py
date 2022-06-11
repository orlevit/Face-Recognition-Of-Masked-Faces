# Desclaimer: This shouldn't be used! tisis in the Git ony because I already writen it, because miss undersatanding and is kept because maybe in te future a suitale case for it wil appear.
# This is remove the pairs in the pairs file that do not appear in the directory(img2pose did not find faces to them).
# The ore corret way is to generate pairs from one of the masked images directorirs so there will be no not exisant pair(one image out of the pair).
import numpy as np
import sys
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import pickle
import os

def write_pairs(pairs_to_write, output_pairs_filename):
    pairs = []
    with open(output_pairs_filename, 'w') as f:
        for one_line in pairs_to_write:
            str_line = '\t'.join(one_line) + '\n'
            f.write(str_line)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def file_name(lfw_dir, pair1, pair2, file_ext):
    file_path = os.path.join(lfw_dir, pair1, pair1 + '_' + '%04d' % int(pair2)+'.'+file_ext)
    if not os.path.exists(file_path):
       file_path = os.path.join(lfw_dir, pair1, '%03d' % int(pair2)+'.'+file_ext)
    if not os.path.exists(file_path):
       file_path = os.path.join(lfw_dir, pair1, pair2 + '.' + file_ext)

    return file_path
def get_paths(lfw_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    exists_pair_list = []
    exists_pair_list.append(pairs[0])
    for pair in pairs[1:]:
        print(pair)
        if len(pair) == 3:
            path0 = file_name(lfw_dir, pair[0], pair[1], file_ext)
            path1 = file_name(lfw_dir, pair[0], pair[2], file_ext)
        elif len(pair) == 4:
            path0 = file_name(lfw_dir, pair[0], pair[1], file_ext)
            path1 = file_name(lfw_dir, pair[2], pair[3], file_ext)

        if os.path.exists(path0) and os.path.exists(path1):
            exists_pair_list.append(pair)
        else:
            nrof_skipped_pairs += 1
            print('not exists', path0, path1)

    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    
    return exists_pair_list

parser = argparse.ArgumentParser(description='Package LFW images')
# general
parser.add_argument('--pairs-file', default='', help='')
parser.add_argument('--checked-dir', default='', help='path of the checkd pairs , what exists and what not.')
parser.add_argument('--output-pairs', default='', help='path of the output pairs after filter the not existing pairs')

args = parser.parse_args()
checked_dir = args.checked_dir
pairs_file = args.pairs_file
output = args.output_pairs

lfw_pairs = read_pairs(pairs_file)
#import pdb;pdb.set_trace();
exists_pair_list = get_paths(checked_dir, lfw_pairs, 'jpg')
write_pairs(exists_pair_list, output)
