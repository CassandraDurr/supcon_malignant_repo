import os
import random

train_data_dir = 'D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_sample/train/1/' 

# Get file count
_, _, files = next(os.walk(train_data_dir))
file_count = len(files)
print(file_count)
# 584 files
# keep 200 files
# randomly delete 384 files

# Number of images in the directory
files = os.listdir(train_data_dir)
for file in random.sample(files,384):
    os.remove(train_data_dir+file)
    
# Get file count
_, _, files = next(os.walk(train_data_dir))
file_count = len(files)
print(file_count)