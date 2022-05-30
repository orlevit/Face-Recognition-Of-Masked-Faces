doc:

Code files:
* all_to_one.py - Combine all the .npy files of the models(in toatl should be 7) to one file (this script is after the dataset npy files were joined together). 
                  This combined the {mask}_all files to one all file(if there were 20 pairs for 7 model it would be the size of (7,2*20*7,512))
                  This could be useful also for the chuncked files that pass through the embedding(for the "composed" model) if is run in the specific directory
                  and remove the endswith "all" in the if condiion.
* all_to_one2.py - The same as "all_to_one.py", but the reading of the files little different. checks who btween the three works for my specific use case.
* all_to_one3.py - The same as "all_to_one.py", but the reading of the files little different. checks who btween the three works for my specific use case.

* joined_test_dbs.py - combined a test files:  getting the embeddings and then combining them. only on the LFW or AGEDB30
* one_bin_to_data.py - create all the bins file for all the modls. It creats embeddings from them and then concatenate them, the training as well as the test files. this is currently not is use
* one_bin_to_data2.py - Same as "one_bin_to_data.py" but saved embedddings in cuncks. maybe this is obselete because of "one_bin_to_data3.py"
* one_bin_to_data3.py - Same as "one_bin_to_data.py" but saved embedddings in cuncks and maybe more correct then "one_bin_to_data2.py"
* one_bin_one_model_to_data.py - Same as "one_bin_to_data3.py", but with the different data location, more easy to change the location. This is the latest

* nd_to_np.py - turn mx.nd array to numpy array and save them.
* all_to_one_one_dir.py - combined all the small ".npy" files of same mask type to one joined file of the same mask.
* all_to_one_one_dir_all.py - combined all the combined files of mask types, to one file in total.


In directory "sbatch":
* sbatch_convert_bin.sh - run "one_bin_to_data3.py" for specific model
* sbatch_convert_bin_models_array.sh - Run "one_bin_to_data3.py" as part of array
* sbatch_one_bin_one_model_to_data.py - Run "one_bin_one_model_to_data.py" in sbatch

General:
Creating the embeddings for the "composed" model:
* one_bin_to_data.py - create directory stracture is in each model there isa directory with the specific dataset pass through that model
* joined_test_dbs.py:
   *  funcion "joined_models" put join all the datasets that passed thrpugh the same model and put them in the directory "db_test_models"
   *  funcion "combine_datasets" comvined all the data from the previoys step into one("db_test_models")
* joined_test_dbs.py & one_bin_to_data.py - are the same except the funcions that are put in comment and the "joined_test_dbs" is for "test" files. The funcion "get_files_loc_models_data" is different as well

Data:
in directory "tmp"(should be renamed) contained data of the LFW embeddings. There are directories for defferent perphese:
	* test - contains the chunks of the all pairs that run throught the models for creating the embeddings. each directory is the data set that contains the models.
	* db_test_models - clusters all the DB that run through one model. create by function "joined_models" in script "one_bin_one_model_to_data.py"
	* db_for_test - opposite of the previous - cluster all the models that run through a DB. This one is for testing a specific occlusion dataset for the "composed" model. created by script "combine_for_test.py"
	* all_test - joined all the data in folder "db_test_models" for training or validation. create by function "combine_datasets" in script "one_bin_one_model_to_data.py"
