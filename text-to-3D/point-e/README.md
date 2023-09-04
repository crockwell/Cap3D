# Generate Test Images using Point-E 

# Usage

1. Install with `pip install -e .`. Please refer to the [original repo](https://github.com/openai/point-e), if any errors occur.

2. Download finetuned checkpoint from https://huggingface.co/datasets/tiange/Cap3D/tree/main/our_finetuned_models, and move it to `model_ckpts`.
```
wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/our_finetuned_models/pointE_finetuned_with_330kdata.pth
mkdir model_ckpts
mv pointE_finetuned_with_330kdata.pth model_ckpts
```

3. execute the below command, the generated mesh will saved at `./pointE_inference/Cap3D_test1'`.
```
python text2ply_pointE.py

# if you need to generate test meshes for the 2k set:
python text2ply_pointE.py --test_type '2k'
```

4. we need to render the generated mesh into images for evaluation.
