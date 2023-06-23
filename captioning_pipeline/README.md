# Captioning Pipeline: Rendering -> BLIP2 -> CLIP -> GPT4
We are still working on the captioning codebase. Full details will be provided soon.

## Rendering
Please first download our Blender from https://huggingface.co/datasets/tiange/Cap3D/blob/main/blender.zip and unzip it at `Cap3D/captioning_pipelin`. then run the below command:
```
# --object_path_pkl: point to a pickle file which store the object path
# --parent_dir: the directory store the rendered images and their associated camera matrix
# Rendered images & camera matrix will stored at partent_dir/Cap3D_imgs/

./blender-3.4.1-linux-x64/blender -b -P render_script.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'
```

