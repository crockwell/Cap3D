# Generate Test Images using Shap-E

## Usage
1. Install with `pip install -e .`. Please make sure you download the shap-E code from this repo as there are modifications compared to the original repo.

2. Download finetuned checkpoint from https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models, and move it to `model_ckpts`.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/our_finetuned_models/shapE_finetuned_with_330kdata.pth
mkdir model_ckpts
mv shapE_finetuned_with_330kdata.pth model_ckpts
```

3. You can then execute step **3.1** to generate rendered images or step **3.2** to first generate meshes and then render them via blender. In our paper, we used step 3.1 to generate images for final scores.

3.1 execute the below command, the generated images will be saved at `./shapE_inference/Cap3D_test1_stf` if `--render_type='stf'` and `./shapE_inference/Cap3D_test1_nerf` if `--render_type='nerf'` (difference between `'stf'` and `'nerf'` can be found in [shapE paper](https://arxiv.org/pdf/2305.02463.pdf).
```
python text2img_shapE.py --save_name 'Cap3D_test1'

# if you need to render images via nerf
python text2img_shapE.py --save_name 'Cap3D_test1' --render_type 'nerf'
```

3.2 Or you can first generate meshes and then render them via blender. First, download our blender, and then run the rendering script. The script will save rendered images at `./args.save_dir/Cap3D_imgs` by loading meshes from `./args.parent_dir+ '_' + args.test_type`.
```
python text2ply_shapE.py --save_name 'Cap3D_test1_meshes'

wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/blender.zip
unzip blender.zip

./blender-3.4.1-linux-x64/blender -b -P render_script_shapE.py -- --test_type 'stf' --save_dir './rendering_output' --parent_dir './shapE_inference/Cap3D_test1_meshes'
```

## Extract Colorful PointClouds
We also provide the code for extracting colorful pointclouds (support Objaverse objects). 

Please run `python extract_pointcloud.py` and the results will be saved at `./extracted_pts`. You can look at the example files to see how to apply it to your own data.

## Citation

if you use shap-E model/data, please cite:
```
@article{jun2023shap,
  title={Shap-e: Generating conditional 3d implicit functions},
  author={Jun, Heewoo and Nichol, Alex},
  journal={arXiv preprint arXiv:2305.02463},
  year={2023}
}
```

If you find our code or data useful, please consider citing:
```
@article{luo2023scalable,
      title={Scalable 3D Captioning with Pretrained Models},
      author={Luo, Tiange and Rockwell, Chris and Lee, Honglak and Johnson, Justin},
      journal={arXiv preprint arXiv:2306.07279},
      year={2023}
}
```
