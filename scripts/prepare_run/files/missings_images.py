import os
import numpy as np
from glob import glob

path1 = '/home/orlev/work/datasets/original_ds/CASIA-WebFace'
img_paths1 = [ "_".join(image.rsplit('.',1)[0].rsplit('/',2)[-2:]) for image  in glob(os.path.join(path1, '**', '*.jpg'), recursive=True)]


path2 = '/home/orlev/work/Face-Recognition-Of-Masked-Faces/images_masked_crop/casia/eyemask'
img_paths2 = [ "_".join(image.rsplit('.',1)[0].rsplit('/',2)[-2:]) for image  in glob(os.path.join(path2, '**', '*.jpg'), recursive=True)]

print(np.setdiff1d(img_paths1, img_paths2))
