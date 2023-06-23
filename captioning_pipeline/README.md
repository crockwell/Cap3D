# Captioning Pipeline: Rendering -> BLIP2 -> CLIP -> GPT4

## Rendering
Please first download blender or use your own blender with enough packages, and then run the render script.

```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/blender.zip
mv blender.zip Cap3D/captioning_pipeline/
unzip blender.zip
```

```
# --object_path_pkl: point to a pickle file which store the object path
# --parent_dir: the directory store the rendered images and their associated camera matrix
# Rendered images & camera matrix will stored at partent_dir/Cap3D_imgs/

./blender-3.4.1-linux-x64/blender -b -P render_script.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```

## BLIP2
Please install BLIP2:
```
conda create -n Cap3D python=3.8
conda activate Cap3D
pip install salesforce-lavis
```

Run the below command, BLIP2 will generate caption for each view (a total of 8) and store at `{parent_dir}/Cap3D_captions`
```
# we use 'pretrain_flant5xxl' in our paper. If it is too big, swtich to 'pretrain_flant5xl'.

python caption_blip2.py --parent_dir ./example_material --model_type 'pretrain_flant5xxl'


# use QA branch to generate geometrical descriptions (as shown in Figure 4, https://arxiv.org/abs//2306.07279)

python caption_blip2.py --parent_dir ./example_material --model_type 'pretrain_flant5xxl' --use_qa
```

## CLIP + GPT4
Please install CLIP and GPT4-api first:
```
pip install openai
pip install git+https://github.com/openai/CLIP.git
```

Please input your openai api key. The results will be saved as `{parent_dir}/Cap3D_captions/Cap3d_captions_final.csv`.
```
python caption_clip_gpt4.py --parent_dir './example_material' --openai_api_key 'Your-key'
```
