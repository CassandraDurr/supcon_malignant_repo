"""The 2020 test data has no labels so we do not have a test set. Create a test set from the labelled training data."""
import os
import random
import shutil

# Set the percentage of images to move from "0" and "1" folders to the test folders
x_percent = 20  # 20% of "0" images
y_percent = 20  # 20% of "1" images

# Directories
train_dir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/train"
test_dir = "D:/Downloads/siim-isic-melanoma-classification/jpeg_adj_final/test"

# Function to count the number of files in a directory
def count_files(directory):
    return len(os.listdir(directory))

# Print the initial counts of images in the train and test directories
print("Before moving files:")
print("Train:")
print("Folder '0':", count_files(os.path.join(train_dir, '0')))
print("Folder '1':", count_files(os.path.join(train_dir, '1')))
print("Test:")
print("Folder '0':", count_files(os.path.join(test_dir, '0')))
print("Folder '1':", count_files(os.path.join(test_dir, '1')))
print()

# Iterate over the "0" and "1" folders within the train directory
for folder_name in ["0", "1"]:
    # Get the list of image files in the current folder
    folder_path = os.path.join(train_dir, folder_name)
    files = os.listdir(folder_path)
    num_files = len(files)
    
    # Calculate the number of files to move based on the specified percentages
    num_files_to_move = int(num_files * (x_percent / 100)) if folder_name == "0" else int(num_files * (y_percent / 100))
    
    # Randomly select files to move
    files_to_move = random.sample(files, num_files_to_move)
    
    # Move the selected files to the corresponding test folder
    for file_name in files_to_move:
        src_path = os.path.join(folder_path, file_name)
        dest_path = os.path.join(test_dir, folder_name, file_name)
        shutil.move(src_path, dest_path)
        
# Print the final counts of images in the train and test directories
print("After moving files:")
print("Train:")
print("Folder '0':", count_files(os.path.join(train_dir, '0')))
print("Folder '1':", count_files(os.path.join(train_dir, '1')))
print("Test:")
print("Folder '0':", count_files(os.path.join(test_dir, '0')))
print("Folder '1':", count_files(os.path.join(test_dir, '1')))

# Before moving files:
# Train:
# Folder '0': 32542
# Folder '1': 584
# Test:
# Folder '0': 0
# Folder '1': 0

# After moving files:
# Train:
# Folder '0': 26034
# Folder '1': 468
# Test:
# Folder '0': 6508
# Folder '1': 116