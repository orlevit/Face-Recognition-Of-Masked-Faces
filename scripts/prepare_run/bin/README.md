doc:
Each pairs file is copied to the images dirctory(maked_faces_cropped_

Code files:
* generate_pairs_mul_dirs.py - generate a pairs when and mix the pairs from multiple masks types(the pairs are with the same mask type) in each fold. the total number of pairs for example : 700000 = 2(match/mismatches) * 7(number of folds) * 25000(pairs of match/mismatch in each fold) afterwards run  "lfw2pack.py".

* generate_pairs3.py - generate pair all from the same mask. actually is writes only the images number without the location from the occluded directory, so in order to create the ".bin" files, it is needed to copy the "pairs.txt" to the masked images folder and run "lfw2pack.py".

* create_bin.py - running sbatchs that create ".bin" files for each mask. this park of the whole of the script frocess that suppose the create all the files before training at one run,
* sbatch_bin.sh - single batch that create a bin file fom the pairs file, with the file "lfw2pack.py"
* lfw2pack.py - create the ".bin" file from the pairs.txt
* lfw.py - help "lfw2pack.py" to read the pairs and get their paths

Data files:
In pairs_files:
pairs_350000.txt - Pairs for same maske type occlusion for the same model(e.g. covid19 masked pair for the covid19 model). 350000 pairs

In bins_files:
* multi_bins - contain the multi bins fils - pairs from mixed(random) masked faces(the pair is withthe same mask type)
* all the "*.bin" in the suitable directory are the result of the "pairs_350000.txt" file. 'mask' model for the same 'mask' dataset.
* In the directories "train" there is the processed data - images form the "bin" file in the suitable directory that were passed throught the suitable masked model.


