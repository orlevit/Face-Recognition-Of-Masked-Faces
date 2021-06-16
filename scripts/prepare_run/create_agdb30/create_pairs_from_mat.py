import scipy.io
import argparse

FOLD_NUMBER = 10
MATCHES_PER_FOLD = 600

def create_pairs_file(mat_file, dest_loc):    
    mat = scipy.io.loadmat(mat_file)
    
    with open(dest_loc, "w") as f:
        f.write(str(FOLD_NUMBER) + "\t" + str(MATCHES_PER_FOLD) + "\n")
        for fold_idx in range(FOLD_NUMBER):
            fold = mat['splits'][fold_idx][0][0][0]

            for match in range(MATCHES_PER_FOLD):
                same_diff_person = fold[1][0][match]

                # First person
                first_person = fold[0][0,match][0][0][0][0]
                first_person_split = first_person.split('_')
                first_person_id = first_person_split[0]
                first_person_picture = first_person_split[2] if first_person_split[2].isnumeric() else first_person_split[3]
                first_person_picture = '{:04d}'.format(int(first_person_picture))

                # Second person        
                second_person = fold[0][1,match][0][0][0][0]
                second_person_split = second_person.split('_')
                second_person_id = second_person_split[0]
                second_person_picture = second_person_split[2] if second_person_split[2].isnumeric() else second_person_split[3]
                second_person_picture = '{:04d}'.format(int(second_person_picture))

                if same_diff_person == 1:
                    f.write(first_person_id + "\t" + first_person_picture + "\t" + second_person_picture + "\n")
                elif same_diff_person == -1:
                    f.write(first_person_id + "\t" + first_person_picture + "\t" + second_person_id + "\t" + second_person_picture + "\n")                
                else:
                    print(f"What is this option? ", same_diff_person)
                    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create pairs file out of agedb mat file')
    parser.add_argument('--mat-file', default='./04_FINAL_protocol_30_years.mat', help='The mat file of matlab with the agedb pairs')
    parser.add_argument('--dest-loc', default='.', help='The location of the pairs files')

# reading the passed arguments
args = parser.parse_args()
mat_file = args.mat_file
dest_loc = args.dest_loc

# Create files
create_pairs_file(mat_file, dest_loc)
