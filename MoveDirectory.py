import shutil
import os

source_dir =  'images/fruits/Test/NotApple'
target_dir = 'images/fruits/Test/NotApple/decompressed'

#source_dir = 'images/fruits/Test/NotApple/decompressed'
#target_dir = 'images/fruits/Test/NotApple'
i = 0

file_names = os.listdir(source_dir)

for listfilename in os.walk(source_dir):
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)