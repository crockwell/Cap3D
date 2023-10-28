import pickle as pkl
import numpy as np
import os
import random
random.seed(0)
from tqdm import tqdm
import torch
import clip
import pandas as pd
import argparse
from torch.nn import CosineSimilarity
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", action="store_true", help="if true, evaluate FID")
    parser.add_argument("--clip_score_precision", action="store_true", help="if true, evaluate CLIP Score and Precision")
    parser.add_argument("--pred_dir", type = str, help="directory with predictions")
    parser.add_argument("--gt_dir", type = str, default='ground_truth', help="directory with ground truth renders")
    parser.add_argument("--embedding_dir", type = str, default='test_embeddings', help="directory to save embeddings")
    parser.add_argument("--test_uid_path", type = str, default='test_uids.pkl', help="pkl with test uids")
    parser.add_argument("--caption_path", type = str, default='our_caption.csv', help="csv with ground truth captions")
    parser.add_argument('--eval_size', type=int, default=2000, help='number of objects to evaluate. We use 300 for DreamField, DreamFusion, 3DFuse')
    return parser.parse_args()

def compute_fid(pred_dir, gt_dir, eval_size, test_uid_path):
    if eval_size < 2000:
        test_uids_total = pkl.load(open(test_uid_path, 'rb'))
        test_uids = test_uids_total[:eval_size]
        if gt_dir[-1] == "/":
            gt_dir = gt_dir[:-1]
        new_gt_dir = gt_dir+"_300"
        if not os.path.exists(new_gt_dir) or len(os.listdir(new_gt_dir)) < eval_size*8:
            print(f"copying {eval_size} ground truth images into new folder for FID calculation")
            os.makedirs(new_gt_dir, exist_ok=True)
            for uid in tqdm(test_uids):
                for j in range(8):
                    os.system("cp "+os.path.join(gt_dir,uid+"_"+str(j)+".jpg")+" "+os.path.join(new_gt_dir,uid+"_"+str(j)+".jpg"))
        gt_dir = new_gt_dir
    
    print("computing FID for",pred_dir,gt_dir)
    os.system('python -m pytorch_fid '+pred_dir+' '+gt_dir)

def get_clip_image_embeddings(pred_dir, eval_size, test_uid_path):
    device = "cuda"
    test_uids_total = pkl.load(open(test_uid_path, 'rb'))
    test_uids = test_uids_total[:eval_size]
    clip_models = {"vit_b_32": "ViT-B/32"}

    for model_name, model_path in clip_models.items():
        print("inference on",model_name)
        model, preprocess = clip.load(model_path, device=device)

        all_features = []
        for uid in tqdm(test_uids):
            all_features_local = []
            for j in range(8):
                img = Image.open(os.path.join(pred_dir,uid+"_"+str(j)+".png")).convert('RGB')
                image = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)

                all_features_local.append(image_features.cpu().numpy()[0])
            
            all_features.append(all_features_local)

        all_features = np.array(all_features)

        with open(os.path.join(pred_dir,f"image_embeddings_{model_name}.npy"),"wb") as f:
            pkl.dump(all_features, f)

def get_clip_text_embeddings(gt_dir, test_uid_path, caption_path):
    device = "cuda"
    test_uids = pkl.load(open(test_uid_path, 'rb'))
    captions = pd.read_csv(caption_path, header=None)

    clip_models = {"vit_b_32": "ViT-B/32"}

    for model_name, model_path in clip_models.items():
        print("inference on",model_name)
        model, _ = clip.load(model_path, device=device)

        all_features = []
        for i in tqdm(range(len(test_uids))):
            caption = captions[captions[0] == test_uids[i]][1].values[0]
            text = clip.tokenize(caption, truncate=True).to(device)

            with torch.no_grad():
                text_features = model.encode_text(text)

            all_features.append(text_features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)

        with open(os.path.join(gt_dir,f"text_embeddings_{model_name}.npy"),"wb") as f:
            pkl.dump(all_features, f)

def compute_clip_score_precision(pred_dir, gt_dir, eval_size, test_uid_path, caption_path):
    cos = CosineSimilarity(dim=1, eps=1e-6)
    clip_models = ["vit_b_32"]

    for model_name in clip_models:
        print("\n model name:", model_name)

        text_embedding_path = os.path.join(gt_dir,f"text_embeddings_{model_name}.npy")
        if not os.path.exists(text_embedding_path):
            # get text clip embeddings
            get_clip_text_embeddings(gt_dir, test_uid_path, caption_path)

        with open(text_embedding_path,"rb") as f:
            text_features = pkl.load(f)

        image_embedding_path = os.path.join(pred_dir,f"image_embeddings_{model_name}.npy")
        if not os.path.exists(image_embedding_path):
            # get image clip embeddings
            get_clip_image_embeddings(pred_dir, eval_size, test_uid_path)
        
        with open(image_embedding_path,"rb") as f: # 
            img_features = pkl.load(f)

        img_features = img_features[:eval_size] 
        text_features = text_features[:eval_size]

        scores = []
        clip_r_1 = []
        clip_r_5 = []
        clip_r_10 = []
        for i in tqdm(range(len(img_features))):
            for j in range(8):
                this_img_feat = torch.from_numpy(img_features[i,j]).unsqueeze(0).cuda()
                this_text_feat = torch.from_numpy(text_features[i]).unsqueeze(0).cuda()
                text_features_cuda = torch.from_numpy(text_features).cuda()

                scores.append((cos(this_img_feat, this_text_feat).cpu().numpy() * 100 * 2.5)[0])

                # similarity between image feature and all text features
                all_local_scores = cos(this_img_feat, text_features_cuda).cpu().numpy() * 100 * 2.5
                
                # see if the image feature is in the top 1, 5, 10
                sorted_scores = np.argsort(-all_local_scores)
                clip_r_1.append(np.where(sorted_scores == i)[0][0] < 1)
                clip_r_5.append(np.where(sorted_scores == i)[0][0] < 5)
                clip_r_10.append(np.where(sorted_scores == i)[0][0] < 10)

        scores = np.array(scores)
        clip_r_1 = np.array(clip_r_1)
        clip_r_5 = np.array(clip_r_5)
        clip_r_10 = np.array(clip_r_10)

        print("CLIP Score", np.round(np.mean(scores),1))
        print("CLIP R@1", np.round(np.mean(clip_r_1)*100,1))
        print("CLIP R@5", np.round(np.mean(clip_r_5)*100,1))
        print("CLIP R@10", np.round(np.mean(clip_r_10)*100,1))

def main():
    args = parse_args()

    if args.clip_score_precision:
        compute_clip_score_precision(args.pred_dir, args.gt_dir, args.eval_size, 
                                     args.test_uid_path, args.caption_path)

    if args.fid:
        compute_fid(args.pred_dir, args.gt_dir, args.eval_size, args.test_uid_path)

if __name__ == "__main__":
    main()
