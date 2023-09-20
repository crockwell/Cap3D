# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 20, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint, fetch_file_cached
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.mesh import TriMesh
import numpy as np
import trimesh
import pickle

import argparse
import random
import time
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', default='pointE_finetuned_with_330kdata.pth', type=str, help="path to finetuned model")
parser.add_argument('--save_name', default='Cap3D_test1', type=str, help="result files save to here")
parser.add_argument('--testset_type', default='2k', type=str, choices=['300','2k'], help="300 or 2k test sets")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('loading finetuned checkpoint: ', opt.ckpt)
base_model.load_state_dict(torch.load(os.path.join('./model_ckpts', opt.ckpt), map_location=device)['model_state_dict'])

### results (.ply) saved here
save_dir = os.path.join('./pointE_inference', opt.save_name)
os.makedirs(save_dir, exist_ok=True)

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()
print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))

batch_size = 1

import pickle
import pandas as pd
test_uids = pickle.load(open('../example_material/test_uids_%s.pkl'%opt.testset_type, 'rb'))
### add the below random command to parallel test
# random.shuffle(test_uids)
captions = pd.read_csv('../example_material/Cap3D_automated_Objaverse.csv', header=None)


print('start mesh generation, generated mesh saved as .ply')
for i in range(len(test_uids)):
    s = time.time()
    ### skip if output mesh exists
    if os.path.exists(os.path.join(save_dir,'%s.ply'%(test_uids[i]))):
       continue
    prompt = captions[captions[0] == test_uids[i]][1].values[0]
    assert captions[captions[0] == test_uids[i]][0].values[0] == test_uids[i]

    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=batch_size, model_kwargs=dict(texts=[prompt,]*batch_size))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]

    mesh = marching_cubes_mesh(
        pc=pc,
        model=model,
        batch_size=4096,
        grid_size=128, 
        progress=True,
    )

    with open(os.path.join(save_dir, '%s.ply'%(test_uids[i])), 'wb') as f:
        mesh.write_ply(f)
    print('mesh generation progress: %d/%d'%(i,len(test_uids)), 'time cost:', time.time()-s)
