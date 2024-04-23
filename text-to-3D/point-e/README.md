# Generate Test Images using Point-E 

## Usage

1. Install with `pip install -e .`. Please refer to the [original repo](https://github.com/openai/point-e), if any errors occur.

2. Download finetuned checkpoint from https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/our_finetuned_models, and move it to `model_ckpts`.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/our_finetuned_models/pointE_finetuned_with_825kdata.pth
mkdir model_ckpts
mv pointE_finetuned_with_825kdata.pth model_ckpts
```

3. execute the below command, the generated mesh will saved at `./pointE_inference/Cap3D_test1`.
```
python text2ply_pointE.py

# if you want to generate test meshes for the 300 set (smaller):
python text2ply_pointE.py --testset_type '300'
```

4. we need to render the generated mesh into images for evaluation. First, download our blender, and then run the rendering script. The script will save rendered images at `./args.save_dir/Cap3D_imgs` by loading meshes from `./args.parent_dir`.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/blender.zip
unzip blender.zip

./blender-3.4.1-linux-x64/blender -b -P render_script_pointE.py -- --save_dir './rendering_output' --parent_dir './pointE_inference/Cap3D_test1'
```

## Citation

if you use point-E model/data, please cite:
```
@article{nichol2022point,
  title={Point-e: A system for generating 3d point clouds from complex prompts},
  author={Nichol, Alex and Jun, Heewoo and Dhariwal, Prafulla and Mishkin, Pamela and Chen, Mark},
  journal={arXiv preprint arXiv:2212.08751},
  year={2022}
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
