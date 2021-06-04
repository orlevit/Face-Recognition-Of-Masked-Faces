import os

from scripts.alignment import run_align
from scripts.create_bin import prerequisite_bin, make_bin
from scripts.create_idx_rec import prerequisite_idx_rec, make_idx_rec
from scripts.script_helper import parse_arguments, run_multy
from scripts.script_config import LFW_PAIRS, CASIA_PAIRS, AGEDB30_PAIRS



def main(args):
    datasets_dirs = [os.path.join(args.input,path) for path in os.listdir(args.input)]

    masked_faces_input_dirs = []
    masked_faces_output_dirs = []
    lfw_dirs = []
    agedb30_dirs = []
    casia_dirs =  []
    for ds_dir in datasets_dirs:
        for masked_dir in os.listdir(ds_dir):
            masked_faces_input_dirs.append(os.path.join(ds_dir, masked_dir))
            masked_faces_output_dirs.append(os.path.join(ds_dir, 'a'+masked_dir))

            if ds_dir.find('/lfw') != -1:
                lfw_dirs.append(os.path.join(ds_dir, 'a'+masked_dir))
            elif ds_dir.find('/agedb') != -1:
                agedb30_dirs.append(os.path.join(ds_dir, 'a'+masked_dir))
            elif ds_dir.find('/casia') != -1:
                casia_dirs.append(os.path.join(ds_dir, 'a'+masked_dir))
            else:
                print(f'What is this dataset?! {ds_dir}')
                # exit(1)

    # run_multy(align_mtcnn, masked_faces_dirs)
    pairs_files =  [[LFW_PAIRS, CASIA_PAIRS, AGEDB30_PAIRS] ,[lfw_dirs, casia_dirs, agedb30_dirs]]

    prerequisite_bin(masked_faces_input_dirs)
    # run_multy(make_bin, masked_faces_output_dirs)
    make_bin(pairs_files)
    prerequisite_idx_rec(casia_dirs)
    # run_multy(make_idx_rec, casia_dirs)


    # waits=[]
    # for i,dir in enumerate(masked_faces_dirs):
    #       full_path, dir_name = os.path.split(os.path.normpath(dir))
    # #     output_dir = os.path.join(full_path, 'a'+dir_name)
    #     wait = len(masked_faces_dirs) - i
    #     waits.append(str(wait))



    print('blablaaaaaaaaaaaaaaaaaaaaaa')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)