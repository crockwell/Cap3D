# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: June 22, 2023
#
# This code is licensed under the MIT License.
#
# If you use or modify this code in your project, we kindly ask that you include
# this copyright notice in each file where the code is used. Your cooperation helps 
# acknowledge the effort that went into creating this project. 
# ==============================================================================

import os
import openai
from IPython import embed
import pickle
import torch
import clip
from PIL import Image
from torch.nn import CosineSimilarity
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--uid_path', type = str, required = True)
parser.add_argument('--openai_api_key', type = str, required = True)
args = parser.parse_args()

# set up API key
openai.api_key = args.openai_api_key

# set up CLIP
cos = CosineSimilarity(dim=1, eps=1e-6)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# set up GPT4
def summarize_captions_gpt4(text):
    prompt = f"Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows: '{text}'. Avoid describing background, surface, and posture. The caption should be:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            top_p = 0.2,
        )
        if response.choices:
            return response.choices[0].message['content']
        else:
            return "No response was received from gpt4 api."
    except Exception as e:
        return f"An error occurred: {str(e)}"

    return response.choices[0].message['content']


# load uid
uids = pickle.load(open('./example_material/%s.pkl'%args.uid_path, 'rb'))

# load captions for each view (8 views in total)
cap0 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view0.pkl', 'rb'))
cap1 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view1.pkl', 'rb'))
cap2 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view2.pkl', 'rb'))
cap3 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view3.pkl', 'rb'))
cap4 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view4.pkl', 'rb'))
cap5 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view5.pkl', 'rb'))
cap6 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view6.pkl', 'rb'))
cap7 = pickle.load(open('./Cap3D_captions/Cap3d_captions_view7.pkl', 'rb'))
caps = [cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7]

output_csv = open('./Cap3D_captions/Cap3d_captions_final.csv', 'w')
writer = csv.writer(output_csv)

print('############begin to generate final captions############')
for i, cur_uid in enumerate(uids):
    cur_captions = []
    cur_final_caption = ''
    # for each view, choose the caption with the highest similarity score
    # run 8 times to get 8 captions
    for k in range(8):
        image = preprocess(Image.open(os.path.join('./Cap3d_captions_view%d'%k, cur_uid + '_%d.png'%k))).unsqueeze(0).to(device)
        cur_caption = caps[k][cur_uid]
        text = clip.tokenize(cur_caption).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        score = cos(image_features, text_features)
        if k == 8-1:
            cur_final_caption += cur_caption[torch.argmax(score)]
        else:
            cur_final_caption += cur_caption[torch.argmax(score)] + ', '

    # sometimes, OpenAI API will return an error, so we need to try again until it works
    while True:
        summary = summarize_captions_gpt4(cur_final_caption)
        if 'An error occurred' not in summary:
            break
    
    print(i, cur_uid, summary)

    # write to csv
    writer.writerow([cur_uid, summary])
    if (i)% 1000 == 0:
        output_csv.flush()
        os.fsync(output_csv.fileno())

output_csv.close()


