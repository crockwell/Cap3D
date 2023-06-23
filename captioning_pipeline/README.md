# Captioning Pipeline: Rendering -> BLIP2 -> CLIP -> GPT4

## Rendering
Please first download our Blender via the below commands. You can use your own Blender, while may need to pip install several packages.

```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/blender.zip
mv blender.zip Cap3D/captioning_pipeline/
unzip blender.zip
```

Please run the below command to render objects (`.glb`, `.obj`) into `.png` images saved at `{parent_dir}/Cap3D_imgs/Cap3D_imgs_view{0~7}/`
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

Please run the below command, BLIP2 will generate caption for each view (a total of 8) and store as `{parent_dir}/Cap3D_captions/Cap3D_captions_view{0~7}.pkl`
```
# --model_type: 'pretrain_flant5xxl' (ours) or 'pretrain_flant5xl' (smaller)

python caption_blip2.py --parent_dir ./example_material --model_type 'pretrain_flant5xxl'


# use QA branch to generate geometrical descriptions (as shown in Figure 4, https://arxiv.org/abs//2306.07279)

python caption_blip2.py --parent_dir ./example_material --model_type 'pretrain_flant5xxl' --use_qa
```

## CLIP + GPT
Please install CLIP and GPT-api first:
```
conda activate Cap3D
pip install openai
pip install git+https://github.com/openai/CLIP.git
```

Please input your openai api key. The results will be saved as `{parent_dir}/Cap3D_captions/Cap3d_captions_final.csv`.
Our paper used GPT4, while you can try GPT3.5 which is much cheaper ($0.03 vs. $0.0015 per 1k tokens): --gpt_type == ['gpt4', 'gpt3.5'].

```
python caption_clip_gpt.py --parent_dir './example_material' --openai_api_key 'Your-key' --gpt_type 'gpt4'
```
