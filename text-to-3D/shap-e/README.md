# Generate Test Images using Shap-E

## Usage
1. Install with `pip install -e .`. Please refer to the [original repo](https://github.com/openai/shap-e), if any errors occur.

2. Download finetuned checkpoint from https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models, and move it to `model_ckpts`.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/our_finetuned_models/shapE_finetuned_with_330kdata.pth
mkdir model_ckpts
mv shapE_finetuned_with_330kdata.pth model_ckpts
```

3. execute the below command, the generated images will saved at `./shapE_inference/Cap3D_test1_stf` if `--render_type='stf'` and `./shapE_inference/Cap3D_test1_nerf` if `--render_type='nerf'` (difference between `'stf'` and `'nerf'` can be found in [shapE paper](https://arxiv.org/pdf/2305.02463.pdf).
```
python text2img_shapE.py 

# if you need to generate images via nerf
python text2ply_pointE.py --render_type 'nerf'
```

## Extract Colorful PointClouds
We also provide the code for extracting colorful pointclouds (support Objaverse objects). Please run `python extract_pointcloud.py`. You can look at the example files to see how to apply it to your own data.

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
