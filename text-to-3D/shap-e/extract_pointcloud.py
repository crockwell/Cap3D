# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 05, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

import torch
import time
import os
from shap_e.util.data_util import load_or_create_multimodal_batch
import argparse
import tqdm
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--uid_path', type = str, default='../../captioning_pipeline/example_material/example_object_path.pkl')
parser.add_argument('--mother_dir', type = str, default='../../captioning_pipeline')
parser.add_argument('--cache_dir', type = str, default='./shapE_cache')
parser.add_argument('--save_name', type = str, default='extracted_pts')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from shap_e.models.download import load_model
xm = load_model('transmitter', device=device)

uid_list = pickle.load(open(args.uid_path, 'rb'))
target_dir = './%s'%args.save_name
os.makedirs(target_dir, exist_ok=True)

with torch.no_grad():  
    for file_path in tqdm.tqdm(uid_list):
        print('Begin to extract point clouds:', file_path)
        try:
            batch = load_or_create_multimodal_batch(
                device,
                model_path= os.path.join(args.mother_dir, file_path),
                mv_light_mode="basic",
                mv_image_size=256,
                pc_num_views=20,
                cache_dir=args.cache_dir,
                verbose=True, # This will show Blender output during renders
            )
            os.popen('python npz2ply.py --npz_filename %s --ply_filename %s'%(os.path.join(args.cache_dir, 'pc_%s_mat_20_524288_16384.npz'%file_path.split('/')[-1]), os.path.join(target_dir, '%s.ply'%file_path.split('/')[-1].split('.')[0]))).read()

        except:
            print('Error:', file_path)
            continue



