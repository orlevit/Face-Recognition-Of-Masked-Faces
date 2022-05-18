doc:

Code files:
* all_to_one.py - Combine all the .npy files of the models(in toatl should be 7) to one file (this script is after the dataset npy files were joined together)
* all_to_one2.py - The same as "all_to_one.py", but the reading of the files little different. checks who btween the three works for my specific use case.
* all_to_one3.py - The same as "all_to_one.py", but the reading of the files little different. checks who btween the three works for my specific use case.

* joined_test_dbs.py - combined a test files:  getting the embeddings and then combining them. only on the LFW or AGEDB30
* one_bin_to_data.py - create all the bins file for all the modls. It creats embeddings from them and then concatenate them, the training as well as the test files. this is currently not is use
* one_bin_to_data2.py - Same as "one_bin_to_data.py" but saved embedddings in cuncks. maybe this is obselete because of "one_bin_to_data3.py"
* one_bin_to_data3.py - Same as "one_bin_to_data.py" but saved embedddings in cuncks and maybe more correct then "one_bin_to_data2.py"
* one_bin_one_model_to_data.py - Same as "one_bin_to_data3.py", but with the different data location, more easy to change the location. This is the latest

* nd_to_np.py - turn mx.nd array to numpy array and save them.

In directory "sbatch":
* sbatch_convert_bin.sh - run "one_bin_to_data3.py" for specific model
* sbatch_convert_bin_models_array.sh - Run "one_bin_to_data3.py" as part of array
* sbatch_one_bin_one_model_to_data.py - Run "one_bin_one_model_to_data.py" in sbatch
