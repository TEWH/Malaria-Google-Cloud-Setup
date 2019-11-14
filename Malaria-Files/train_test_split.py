import os
import random
from shutil import copyfile
from zipfile import ZipFile

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CELL_IMAGES_DIR = os.path.join(ROOT_DIR, "cell_images")
PARASITIZED_DIR = os.path.join(CELL_IMAGES_DIR, "Parasitized")
UNINFECTED_DIR = os.path.join(CELL_IMAGES_DIR, "Uninfected")
MALARIA_DATASET_DIR = os.path.join(ROOT_DIR, "Malaria_Dataset")
TRAINING_SET_DIR = os.path.join(MALARIA_DATASET_DIR, "Training_Set")
TRAIN_PARASITIZED_DIR = os.path.join(TRAINING_SET_DIR, "Parasitized")
TRAIN_UNINFECTED_DIR = os.path.join(TRAINING_SET_DIR, "Uninfected")
TESTING_SET_DIR = os.path.join(MALARIA_DATASET_DIR, "Testing_Set")
TEST_PARASITIZED_DIR = os.path.join(TESTING_SET_DIR, "Parasitized")
TEST_UNINFECTED_DIR = os.path.join(TESTING_SET_DIR, "Uninfected")

# Ignore script if test/train set already exists
if os.path.isdir(MALARIA_DATASET_DIR):
    quit()

# Extract images if not already extracted
if not os.path.isdir("cell_images"):
    print("Extracting images...")

    with ZipFile(os.path.join(ROOT_DIR, "cell_images.zip"), "r") as zipObj:
        zipObj.extractall()

cell_images = os.listdir(CELL_IMAGES_DIR)
parasitized_images = os.listdir(PARASITIZED_DIR)
uninfected_images = os.listdir(UNINFECTED_DIR)
train_test_ratio = 0.8
target_parasitized_train_size = int(len(parasitized_images) * train_test_ratio)
target_parasitized_test_size = len(parasitized_images) - target_parasitized_train_size
target_uninfected_train_size = int(len(uninfected_images) * train_test_ratio)
target_uninfected_test_size = len(uninfected_images) - target_uninfected_train_size
target_test_size = target_parasitized_test_size + target_uninfected_test_size
target_train_size = target_parasitized_train_size + target_uninfected_train_size

# Randomly move 20% of parisitized images to testing set
print("Copying parisitized images to testing set...")
os.makedirs(TEST_PARASITIZED_DIR, exist_ok=True)

while len(parasitized_images) > target_parasitized_test_size:
    cell_image = random.choice(parasitized_images)
    cell_image_dir = os.path.join(PARASITIZED_DIR, cell_image)
    renamed_dir = os.path.join(TEST_PARASITIZED_DIR, cell_image)

    copyfile(cell_image_dir, renamed_dir)
    parasitized_images.remove(cell_image)

# Move the remaining parisitized images to training set
print("Copying parisitized images to training set...")
os.makedirs(TRAIN_PARASITIZED_DIR, exist_ok=True)

while len(parasitized_images) > 0:
    cell_image = random.choice(parasitized_images)
    cell_image_dir = os.path.join(PARASITIZED_DIR, cell_image)
    renamed_dir = os.path.join(TRAINING_SET_DIR, "Parasitized", cell_image)

    copyfile(cell_image_dir, renamed_dir)
    parasitized_images.remove(cell_image)

# Randomly move 20% of uninfected images to testing set
print("Copying uninfected images to testing set...")
os.makedirs(TEST_UNINFECTED_DIR, exist_ok=True)

while len(uninfected_images) > target_uninfected_test_size:
    cell_image = random.choice(uninfected_images)
    cell_image_dir = os.path.join(UNINFECTED_DIR, cell_image)
    renamed_dir = os.path.join(TESTING_SET_DIR, "Uninfected", cell_image)

    copyfile(cell_image_dir, renamed_dir)
    uninfected_images.remove(cell_image)

# Move the remaining uninfected images to training set
print("Copying uninfected images to training set...")
os.makedirs(TRAIN_UNINFECTED_DIR, exist_ok=True)

while len(uninfected_images) > 0:
    cell_image = random.choice(uninfected_images)
    cell_image_dir = os.path.join(UNINFECTED_DIR, cell_image)
    renamed_dir = os.path.join(TRAINING_SET_DIR, "Uninfected", cell_image)

    copyfile(cell_image_dir, renamed_dir)
    uninfected_images.remove(cell_image)

print("Done!")
