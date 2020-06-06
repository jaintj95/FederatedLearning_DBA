import glob
import os
import platform
import zipfile
from shutil import move
from os import listdir, rmdir

source_dir = "tiny-imagenet-200.zip"
target_folder = '../data/tiny-imagenet-200/val/'


# Extract tiny imagenet zip
with zipfile.ZipFile(source_dir, 'r') as zip_ref:
    if not os.path.exists("../data"):
        os.mkdir("../data")
    zip_ref.extractall("../data")

val_dict = {}
with open(target_folder + 'val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob(target_folder + 'images/*.JPEG')

for path in paths:
    if platform.system() == 'Windows':
        file = path.split('\\')[-1]
    else:
        file = path.split('/')[-1]
        
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))

    dest = target_folder + str(folder) + '/' + str(file)
    move(path, dest)

# for path in paths:
#     if platform.system() == 'Windows':
#         file = path.split('\\')[-1]
#     else:
#         file = path.split('/')[-1]
#     folder = val_dict[file]
#     dest = target_folder + str(folder) + '/' + str(file)
#     move(path, dest)

os.remove('../data/tiny-imagenet-200/val/val_annotations.txt')
rmdir('../data/tiny-imagenet-200/val/images')
print('Successfully reformatted the validation images')