This folder contains codes for our Text-to-3D experiments in terms of finetuning and evaluation. 

```
# finetune Point-E
# first git clone https://github.com/openai/point-e.git, and install with pip install -e .
# move finetune_pointE.py and corresponding files in example_material to point-e directory
# need to modify pts_pt_path to the directory you store the point clouds .pt files (download from https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips)
# you may also change the caption and train/val set file path in line #48,49,68,69

python finetune_pointE.py --save_name 'pointE_bs64_lr1e5' --pts_pt_path './Cap3D_pcs_pt'
```