import os
from prepare_run.bin.create_bin import make_bin, prerequisite_bin
from prepare_run.align.alignment import make_align
from prepare_run.idx_rec.create_idx_rec import make_idx_rec, prerequisite_idx_rec
from script_helper import parse_arguments
from script_config import LFW_PAIRS, CASIA_PAIRS, AGEDB30_PAIRS


def main(args):
    datasets_dirs = [os.path.join(args.input, path) for path in os.listdir(args.input)]

    lfw_dirs = []
    casia_dirs = []
    agedb30_dirs = []
    masked_faces_input_dirs = []
    masked_faces_input_dirs_aligned = []

    # Get the directories
    for ds_dir in datasets_dirs:
        if  ds_dir.endswith('casia'):
            for masked_dir in os.listdir(ds_dir):# and (not masked_dir.endswith('casia')):
                if not masked_dir.startswith('a') and (masked_dir.endswith('scmask') or masked_dir.endswith('shmask')):
                    # masked_faces_input_dirs.append(os.path.join(ds_dir, masked_dir))
                    if ds_dir.find('/lfw') != -1:
                        lfw_dirs.append(os.path.join(ds_dir, 'a' + masked_dir))
                        masked_faces_input_dirs_aligned.append(os.path.join(ds_dir, 'a' + masked_dir))
                        masked_faces_input_dirs.append(os.path.join(ds_dir, masked_dir))
                    elif ds_dir.find('/agedb') != -1:
                        agedb30_dirs.append(os.path.join(ds_dir, 'a' + masked_dir))
                        masked_faces_input_dirs_aligned.append(os.path.join(ds_dir, 'a' + masked_dir))
                        masked_faces_input_dirs.append(os.path.join(ds_dir, masked_dir))
                    elif ds_dir.find('/casia') != -1:
                        casia_dirs.append(os.path.join(ds_dir, 'a' + masked_dir))
                        masked_faces_input_dirs_aligned.append(os.path.join(ds_dir, 'a' + masked_dir))
                        masked_faces_input_dirs.append(os.path.join(ds_dir, masked_dir))
                    else:
                        print(f'What is this dataset?! {ds_dir} and dir name; {masked_dir}')
    print(masked_faces_input_dirs)
    # Align
    #make_align(masked_faces_input_dirs)
    pairs_files = [[LFW_PAIRS, CASIA_PAIRS, AGEDB30_PAIRS], [lfw_dirs, casia_dirs, agedb30_dirs]]

    # Bin
    #prerequisite_bin(pairs_files)
    #make_bin(masked_faces_input_dirs_aligned)

    # IDX & REC
    prerequisite_idx_rec(casia_dirs)
    make_idx_rec(casia_dirs)

    print('Finished!')


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
