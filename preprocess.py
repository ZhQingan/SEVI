import os
import json
import argparse
from tqdm import tqdm
import random
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# #####--------两种方法都可以
# processor = AutoProcessor.from_pretrained("/path/to/openai/clip-vit-large-patch14-336",local_files_only=True)
# model = CLIPModel.from_pretrained("/path/to/openai/clip-vit-large-patch14-336",local_files_only=True).to(device)
# model.eval()

def get_features(pt):
    image = Image.open(pt)
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        image_features = model.get_image_features(**inputs)
    return image_features[0]

def build_embedding(imgs, pt):
    features = []
    for img in (imgs):
        features.append((img, get_features(img)))

    torch.save(features, pt)

parser = argparse.ArgumentParser()
parser.add_argument("--image_file", type=str, default="")
parser.add_argument("--data_file", type=str, default="")
parser.add_argument("--output_file", type=str, default="")
parser.add_argument("--retrieve", type=bool, default=False)
args = parser.parse_args()

imgs = os.listdir(args.image_file)
data = [json.loads(_) for _ in open(args.data_file).readlines()]
outer=open(args.output_file, 'w', encoding='utf-8')

for meta in tqdm(data):
    image = meta['image']
    if args.retrieve is False: # Random Image Selection
        t =random.randint(0,len(imgs))
        meta['another_imame']=imgs[t]
    else:
        
        if os.path.exists("embedding.pt"):
            embeds = torch.load("embedding.pt", map_location='cpu')
        else:
            build_embedding(imgs, "embedding.pt")
            embeds = torch.load("embedding.pt", map_location='cpu')
        sim_min = 1
        img_min = None
        img_emb = get_features(os.path.join(args.image_file,image))
        for img, emb in embeds:
            sim = F.cosine_similarity(emb, img_emb, 0, 1e-8).item()
            sim = (1+sim)/2
            if sim < sim_min:
                sim_min = sim
                img_min = img

        meta['another_imame']=img_min
    outer.write(json.dumps(meta,ensure_ascii=False)+'\n')