import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import glob
import pickle as pkl
from tqdm import tqdm
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type = str, default='./example_material')
    parser.add_argument("--model_type", type = str, default='pretrain_flant5xxl', choices=['pretrain_flant5xxl', 'pretrain_flant5xl'])
    parser.add_argument("--use_qa", action="store_true")
    return parser.parse_args()

def main(view_number):
    args = parse_args()

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    all_output = {}

    name = 'blip2_t5'
    model_type = args.model_type

    outfilename = f'{args.parent_dir}/Cap3D_captions/Cap3D_captions_view{view_number}.pkl'
    infolder = f'{args.parent_dir}/Cap3D_imgs/Cap3D_imgs_view{view_number}/*.png'
    
    if os.path.exists(outfilename):
        with open(outfilename, 'rb') as f:
            all_output = pkl.load(f)

    print("number of annotations so far",len(all_output))

    model, vis_processors, _ = load_model_and_preprocess(name=name, model_type=model_type, is_eval=True, device=device)
    ct = 0

    all_files = glob.glob(infolder)
    all_imgs = [x for x in all_files if ".png" in x.split("_")[-1]]
    print("len of .png", len(all_imgs))

    all_imgs = [x for x in all_imgs if x not in all_output]
    print("len of new", len(all_imgs))

    for filename in tqdm(all_imgs):
        # skip the images we have already generated captions
        if os.path.exists(outfilename):
                if os.path.basename(filename).split('.')[0] in all_output.keys():
                    continue
        try:
            raw_image = Image.open(filename).convert("RGB")
        except:
            print("file not work skipping", filename)
            continue

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        if args.use_qa:
            prompt = "Question: what object is in this image? Answer:"
            object = model.generate({"image": image, "prompt": prompt})[0]
            full_prompt = "Question: what is the structure and geometry of this %s?" % object
            x = model.generate({"image": image, "prompt": full_prompt}, use_nucleus_sampling=True, num_captions=5)
        else:
            x = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=5)

        all_output[os.path.basename(filename).split('.')[0]] = [z for z in x]
        
        if ct < 10 or (ct % 100 == 0 and ct < 1000) or (ct % 1000 == 0 and ct < 10000) or ct % 10000 == 0:
            print(filename)
            print([z for z in x])

            with open(outfilename, 'wb') as f:
                pkl.dump(all_output, f)
            
        ct += 1

    with open(outfilename, 'wb') as f:
        pkl.dump(all_output, f)

if __name__ == "__main__":
    for i in range(8):
        main(view_number=i)
