This folder contains codes for our Text-to-3D experiments in terms of finetuning and evaluation. 

Our finetuned models can be downloaded from https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models. It currently includes Shap-E and Point-E (text-to-3D) models fintuned with 330k Cap3D captions. 

## Finetune Point-E (Text-to-3D)
```
# first git clone https://github.com/openai/point-e.git, and install with pip install -e .
# move finetune_pointE.py and corresponding files in example_material to point-e directory
# need to modify --pts_pt_path to the directory you store the point clouds .pt files (download from https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips)
# you may also change the caption and train/val set file path in line #48,49,68,69

python finetune_pointE.py --gpus 4 --batch_size 64 --save_name 'pointE_bs64_lr1e5' --pts_pt_path './Cap3D_pcs_pt'
```

## Finetune Shap-E
```
# first git clone https://github.com/openai/shap-e, and install with pip install -e .
# move finetune_shapE.py and corresponding files in example_material to shap-e directory
# need to modify --latent_code_path to the directory you store the shap-E latent code .pt files (download from https://huggingface.co/datasets/tiange/Cap3D/tree/main/ShapELatentCode_zips)
# you may also change the caption and train/val set file path in line #48,49,68,69

python finetune_shapE.py --gpus 4 --batch_size 16 --save_name 'shapE_bs16_lr1e5' --latent_code_path './Cap3D_latentcodes'
```
