import os

from scripts.alignment import align_mtcnn
from scripts.script_helper import parse_arguments, run_align



def main(args):
    datasets_dirs = [os.path.join(args.input,path) for path in os.listdir(args.input)]
    masked_faces_dirs = [os.path.join(ds_dir, masked_dir) for ds_dir in datasets_dirs for masked_dir in os.listdir(ds_dir)]
    align_mtcnn(masked_faces_dirs)
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