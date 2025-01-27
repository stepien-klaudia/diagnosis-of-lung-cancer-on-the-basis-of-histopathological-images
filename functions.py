import os
import shutil
import random
import csv

def copy_random_data_from_one_folder(source_path, dest_path):
    all_files = os.listdir(source_path)
    random.shuffle(all_files)
    for i, file in enumerate(all_files):
      if i % 3 == 0:
        source_file = os.path.join(source_path, file)
        destination_file = os.path.join(dest_path, file)
        shutil.copy2(source_file, destination_file)

def copy_random_data_from_all_folders():
   copy_random_data_from_one_folder('pliki\Testing\Lung_adenocarcinoma','pliki_new\Testing\Lung_adenocarcinoma')
   copy_random_data_from_one_folder('pliki\Testing\Lung_benign_tissue','pliki_new\Testing\Lung_benign_tissue')
   copy_random_data_from_one_folder('pliki\Testing\Lung_squamous cell_carcinoma','pliki_new\Testing\Lung_squamous cell_carcinoma')
   copy_random_data_from_one_folder('pliki\Training\Lung_adenocarcinoma','pliki_new\Training\Lung_adenocarcinoma')
   copy_random_data_from_one_folder('pliki\Training\Lung_benign_tissue','pliki_new\Training\Lung_benign_tissue')
   copy_random_data_from_one_folder('pliki\Training\Lung_squamous cell_carcinoma','pliki_new\Training\Lung_squamous cell_carcinoma')

def delete_all_test_jpg_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                os.remove(os.path.join(root, file))  
         
   


