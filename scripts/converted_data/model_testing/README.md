Doc:

test.py - 
Test the latest "composed" model, it run on all the LFW images(size of (7,12000,512), first dimension is the model type, the second is the pairs - two images for, one after the other - so in lFW 6000 pairs * 2 images per pairs is 12000. the thied is the embeddings size)
The threshold for each two images is 0, above it the same, below they are different person.

18/5/22
currently it is model: 350000_pairs_batch_all_hidden4096_NeuralNetwork5_lr1e-05_32_D20_02_2022_T18_53_58_770221.pt 
The data:
The same mask type on deffernet models.
files are in: /home/orlev/work/Face-Recognition-Of-Masked-Faces/scripts/converted_data/ready_data/350000_test_lfw_casia_pairs
