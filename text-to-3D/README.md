# Text-to-3D codes for finetuning and evaluation
This folder contains codes for our Text-to-3D experiments in terms of finetuning and evaluation. 

Our finetuned models can be downloaded from https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models. It currently includes Shap-E and Point-E (text-to-3D) models finetuned with 330k Cap3D captions. 

## Finetuning

### Finetune Point-E (Text-to-3D)
```
# first git clone https://github.com/openai/point-e.git, and install with pip install -e .
# move finetune_pointE.py and corresponding files in example_material to point-e directory
# need to modify --pts_pt_path to the directory you store the point clouds .pt files (download from https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips)
# you may also change the caption and train/val set file path in line #48,49,68,69

python finetune_pointE.py --gpus 4 --batch_size 64 --save_name 'pointE_bs64_lr1e5' --pts_pt_path './Cap3D_pcs_pt'
```

### Finetune Shap-E
```
# first git clone https://github.com/openai/shap-e, and install with pip install -e .
# move finetune_shapE.py and corresponding files in example_material to shap-e directory
# need to modify --latent_code_path to the directory you store the shap-E latent code .pt files (download from https://huggingface.co/datasets/tiange/Cap3D/tree/main/ShapELatentCode_zips)
# you may also change the caption and train/val set file path in line #48,49,68,69

python finetune_shapE.py --gpus 4 --batch_size 16 --save_name 'shapE_bs16_lr1e5' --latent_code_path './Cap3D_latentcodes'
```

## Evaluation

### Generate Test Images using Point-E (Text-to-3D)
```
Coming soon!
```

### Evaluate Generated Images on Test Set
```
# Step 1. Run 'pip install git+https://github.com/openai/CLIP.git' and 'pip install pytorch-fid'
# Step 2. Obtain test uids, captions, and ground truth test images from HuggingFace
# as test_uids.pkl, Cap3D_automated_Objaverse.csv, and test_images/
# Step 3. Obtain renders synthesized from a model.

# Basic Script
python evaluate.py --clip_score_precision --fid --eval_size 2000 --gt_dir test_images/ --pred_dir pointe_text_to_3d_preds/ --test_uid_path test_uids.pkl --caption_path Cap3D_automated_Objaverse.csv

# For Point-E, Shap-E set eval_size as 2000, for DreamField, DreamFusion, 3DFuse set eval_size as 300
# Must select only one of --fid or --clip_score_precision in a given call unless have multiple GPUs available. 
```
