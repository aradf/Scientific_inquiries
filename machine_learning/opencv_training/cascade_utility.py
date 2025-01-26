import os
import shutil

# change the working directory to the folder this script is in.
# Doing this since the images from the video will be captured in this folder
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )
cascade_directory = os.path.join( os.path.dirname(__file__), 
                                 'cascade')

def generate_negativeDescription():
    # open the ouputfile for writing.  Will overwrite all existing dta in there.
    negative_file = os.path.join( cascade_directory, 'neg.txt')
    with open(negative_file,'w') as neg_filename:
        # loop over all the file names.
        file_name = os.path.join( cascade_directory, 'negative') 
        for filename in os.listdir(file_name):
            neg_filename.write(os.path.join('negative', filename + '\n'))
    print("done ...")

def sort_fileName(file_name = None, type=None, init_counter=None):
    # sort and name all file names
    print("sort_fileName")            
    file_name = os.path.join( cascade_directory, type)
    counter = init_counter 
    directory_files = sorted(os.listdir(file_name))
    for source_filename in directory_files:
        first = os.path.join( file_name, source_filename)
        second = os.path.join( file_name, 
                               'out_put' + str(counter) + '.png')
        counter = counter + 1
        shutil.move( first, 
                     second)
    print("done ...")

# __name__
if __name__ == "__main__":
    file_name = os.path.join( cascade_directory, 'negative')
    sort_fileName(file_name=file_name, type='negative', init_counter=1000)
    generate_negativeDescription()

    file_name = os.path.join( cascade_directory, 'positive')
    sort_fileName(file_name=file_name, type='positive', init_counter=2000)
    print("done ...")

    ### opencv_annotation utility for the pos.txt
