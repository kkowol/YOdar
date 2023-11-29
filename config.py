import numpy as np
import os
#=========== 1. Stage: Data preparation ================#
### file: preprep.py

### working directory
path                 = str(os.getcwd())
data_path            = ''   #TODO: insert absolut path for data
CUDA_VISIBLE_DEVICES = '0'

### setup location and condition
locations            = ['Boston', 'Singapore', 'All']
conditions           = ['day', 'rain', 'night', 'all']
name_affix           = [locations[2], conditions[3]]

conditions_night     = ['All', 'all']
nr_night_scenes      = 25   #number between [0,99]

classes              = ['vehicle', 'all']
pref_class           = classes[0]

slices               = 160
use_sweeps           = False # additional data

### get Information about nuScenes Database and save/load all tokens
save_tokens         = True
load_tokens         = True

### choose required data
save_training_data  = True  # data for training (CNN)
get_input_matrix    = True  # radar data for YOdar (Framework for object detection with YOLO and Radar)
get_kitti           = True  # get data in KITTI format

#=========== 2. Stage: training radar network ==========#
### file: training_radar.py

batch_size          = 128
weightdecay         = 3e-4
epochs              = 20
lr                  = 0.001
name_affix_train    = [locations[2], conditions[3]]
filename_end        = f'_{name_affix_train[0]}_{name_affix_train[1]}.npy'
try:
    x_train             = np.load(path + f'/data_{slices}/x_train' + filename_end)
    y_train             = np.load(path + f'/data_{slices}/y_train' + filename_end)
    x_val               = np.load(path + f'/data_{slices}/x_val' + filename_end)
    y_val               = np.load(path + f'/data_{slices}/y_val' + filename_end)
    x_test              = np.load(path + f'/data_{slices}/x_test' + filename_end)
    y_test              = np.load(path + f'/data_{slices}/y_test' + filename_end)
except:
    pass
#=========== 3. Stage: YOLO ===================#
### open folder YOLO
# 01. open ./core/config.py and adjust parameters if necessary
# 02. start ./YOLO/train.py
# 03. enter the desired checkpoint in ./core/config.py 
# 04. start ./YOLO/evaluate_radar.py 
# 05. start vis.py to get diagrams
