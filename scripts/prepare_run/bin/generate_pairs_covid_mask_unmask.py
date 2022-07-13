# This version relay that the picures are in lfw or casia format.
# This verson creats the pairs with the picture number(instead presume that they are ordered from 1..number_of_picture_in_dir)
# the main change are in lines 63,64.86 and 87 by change function _clean_images in line 99

import math
import glob
import io
import os
import random
from argparse import ArgumentParser, Namespace
from typing import List, Optional, Set, Tuple, cast
import itertools
import numpy as np
from glob import iglob
from multiprocessing import Pool
import pathlib
from datetime import datetime

Mismatch = Tuple[str, int, str, int]
Match = Tuple[str, int, int]
CommandLineArgs = Namespace
random.seed(9)

def write_pairs_to_file(fname: str,
                        match_folds: List[List[Match]],
                        mismatch_folds: List[List[Mismatch]],
                        num_folds: int,
                        num_matches_mismatches: int) -> None:
    metadata = '{}\t{}\n'.format(num_folds, num_matches_mismatches)
    with io.open(fname,
                 'w',
                 io.DEFAULT_BUFFER_SIZE,
                 encoding='utf-8') as fpairs:
        fpairs.write(metadata)
        for match_fold, mismatch_fold in zip(match_folds, mismatch_folds):
            for match in match_fold:
                line = '{}\t{}\t{}\n'.format(match[0], match[1], match[2])
                fpairs.write(line)
            for mismatch in mismatch_fold:
                line = '{}\t{}\t{}\n'.format(
                    mismatch[0], mismatch[1], mismatch[2])
                fpairs.write(line)
        fpairs.flush()


def _split_people_into_folds(image_dir: str,
                             num_folds: int) -> List[List[str]]:
    names = [d for d in os.listdir(image_dir)
             if os.path.isdir(os.path.join(image_dir, d))]
    random.shuffle(names)
    return [list(arr) for arr in np.array_split(names, num_folds)]


def _make_matches(image_dir: str,
                  format_style: str,
                  people: List[str],
                  total_matches: int) -> List[Match]:
    matches = cast(Set[Match], set())
    curr_matches = 0
    while curr_matches < total_matches:
        print(f'Match numbrt: {curr_matches}')
        person = random.choice(people)
        images = _clean_images(os.path.join(image_dir, 'acovid19mask'), str(person), format_style)
        if len(images) > 1:
            img1, img2 = sorted(
                [int(random.choice(images)),
                 int(random.choice(images))])
            img1_w_path = os.path.join(image_dir, 'anomask', str(person), str(img1))
            img2_w_path = os.path.join(image_dir, 'acovid19mask', str(person), str(img1))
            match = (img1_w_path, img2_w_path, 1)
            if (img1 != img2) and (match not in matches):
                matches.add(match)
                curr_matches += 1
    return sorted(list(matches), key=lambda x: x[0].lower())


def _make_mismatches(image_dir: str,
                     format_style: str,
                     people: List[str],
                     total_matches: int) -> List[Mismatch]:
    mismatches = cast(Set[Mismatch], set())
    mismatches_not_to_write = cast(Set[Mismatch], set())

    curr_matches = 0
    while curr_matches < total_matches:
        print(f'Mismatch numbrt: {curr_matches}')
        person1 = random.choice(people)
        person2 = random.choice(people)
        if person1 != person2:
            person1_images = _clean_images(os.path.join(image_dir, 'acovid19mask'), person1, format_style)
            person2_images = _clean_images(os.path.join(image_dir, 'acovid19mask'), person2, format_style)
            if person1_images and person2_images:
                img1 = int(random.choice(person1_images))
                img2 = int(random.choice(person2_images))
                if person1.lower() > person2.lower():
                    person1, img1, person2, img2 = person2, img2, person1, img1
                mismatch = (person1, img1, person2, img2)
                img1_w_path = os.path.join(image_dir, 'anomask', str(person1), str(img1))
                img2_w_path = os.path.join(image_dir, 'acovid19mask', str(person2), str(img2))
                mismatch = (img1_w_path, img2_w_path, 0)
                if mismatch not in mismatches_not_to_write:
                    mismatches.add(mismatch)
                    mismatches_not_to_write.add(mismatch)
                    mismatches_not_to_write.add((person2, img2, person1, img1))
                    curr_matches += 1
    return sorted(list(mismatches), key=lambda x: x[0].lower())


def _clean_images(base: str, folder: str, format_type: str):
    images = os.listdir(os.path.join(base, folder))
    if format_type == "lfw":
       images = [image.rsplit('_',1)[1].rsplit('.',1)[0] for image in images if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg")]
    else: #format_type == "casia" 
       images = [image.rsplit('.',1)[0] for image in images if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg")]
    return images


def generate_pairs(
        image_dir: str,
        format_style: str,
        num_folds: int,
        num_matches_mismatches: int,
        write_to_file: bool=False,
        pairs_file_name: str="") -> None:

    print('aaaaaaaaaaaaaaaaaaa')
    tic = datetime.now()
    people_folds = _split_people_into_folds(os.path.join(image_dir, 'acovid19mask'), num_folds)
    toc = datetime.now()
    print('people_folds')
    print('Loading all people', (toc-tic).total_seconds()/60)
    matches = []
    mismatches = []
    if num_matches_mismatches == -1:
        min_match_mismatch_pairs = math.inf
        # find number of minimum matches-mismatches(minimum is calculated because the folds should be equal)
        # folder must be non empty
        for fold_i, fold in enumerate(people_folds):
            print(f'Generate folder:({fold_i}/{len(people_folds)})')
            minimum_num_of_matches_fold = 0
            for people in fold:
                num_of_img_per_person = _clean_images(image_dir, people, format_style)
                minimum_num_of_matches_fold += int(len(list(itertools.combinations(num_of_img_per_person, 2)))/2)

            if minimum_num_of_matches_fold < min_match_mismatch_pairs:
                min_match_mismatch_pairs = minimum_num_of_matches_fold

        print("Number of max match mismatch pairs is: ", min_match_mismatch_pairs)
        num_matches_mismatches = min_match_mismatch_pairs

    for fold_i, fold in enumerate(people_folds):
        print(f'Generate folder:({fold_i}/{len(people_folds)})')
        matches.append(_make_matches(image_dir, 
                                     format_style,
                                     fold,
                                     num_matches_mismatches))
        mismatches.append(_make_mismatches(image_dir,
                                           format_style,
                                           fold,
                                           num_matches_mismatches))
    if write_to_file:
        write_pairs_to_file(pairs_file_name,
                            matches,
                            mismatches,
                            num_folds,
                            num_matches_mismatches)
    return matches, mismatches


def _cli() -> None:
    args = _parse_arguments()
    generate_pairs(
        args.image_dir,
        args.format_style,
        args.num_folds,
        args.num_matches_mismatches,
        True,
        args.pairs_file_name)


def _parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--image_dir',
                        type=str,
                        required=True,
                        help='Path to the image directory.')
    parser.add_argument('--format_style',
                        type=str,
                        required=True,
                        help='This should be the string "lfw" or "casia"')
    parser.add_argument('--pairs_file_name',
                        type=str,
                        required=True,
                        help='Filename of pairs.txt')
    parser.add_argument('--num_folds',
                        type=int,
                        required=True,
                        help='Number of folds for k-fold cross validation.')
    parser.add_argument('--num_matches_mismatches',
                        type=int,
                        required=False,
                        default=-1,
                        help='Number of matches/mismatches per fold.')
    return parser.parse_args()


if __name__ == '__main__':
    _cli()
