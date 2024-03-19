# Captioning Pipeline: Rendering -> BLIP2 -> CLIP -> GPT4

This directory hosts our captioning pipeline code, which involves (1) rendering 3D objects into eight views, (2) generating five captions per view with BLIP2, (3) selecting one caption per view using CLIP, and (4) consolidating a final caption from multi-view using GPT4. Example files in `./example_material` provide guidance on running our pipeline using the below commands.  You should be able to generate the final captions for the [ten example objects](https://github.com/crockwell/Cap3D/tree/main/captioning_pipeline/example_material/glbs).

## Rendering
Please move to `Cap3D/captioning_pipeline/` and download our Blender via the below commands. You can use your own Blender, while may need to pip install several packages.

```
git clone https://github.com/crockwell/Cap3D.git
cd Cap3D/captioning_pipeline/ 

wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
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
# --model_type: 'pretrain_flant5xxl' (used in our paper) or 'pretrain_flant5xl' (smaller)

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
