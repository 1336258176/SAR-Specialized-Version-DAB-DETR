import json
import numpy as np
import argparse
from sklearn.cluster import KMeans
import os

def parse_args():
    parser = argparse.ArgumentParser("K-Means for DAB-Deformable-DETR prior boxes")
    parser.add_argument('--anno_file', type=str, required=True, help='Path to COCO instances_train.json')
    parser.add_argument('--num_clusters', type=int, default=300, help='Number of prior boxes (queries)')
    parser.add_argument('--save_path', type=str, default='tools/priors/hrsid_priors_300.npy', help='Save path for priors')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Loading annotations from {args.anno_file}")
    with open(args.anno_file, 'r') as f:
        data = json.load(f)
    
    img_id_to_dim = {img['id']: (img['width'], img['height']) for img in data['images']}
    
    normalized_boxes = []
    for ann in data['annotations']:
        if 'bbox' not in ann:
            continue
        x, y, w, h = ann['bbox']
        img_w, img_h = img_id_to_dim[ann['image_id']]
        
        # normalize to [0, 1]
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        
        normalized_boxes.append([cx, cy, nw, nh])
    
    normalized_boxes = np.array(normalized_boxes)
    print(f"Total boxes extracted: {len(normalized_boxes)}")
    print(f"Running K-Means with k={args.num_clusters}...")
    
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init=10)
    kmeans.fit(normalized_boxes)
    
    priors = kmeans.cluster_centers_
    # clip to eps to avoid log(0) in inverse_sigmoid
    eps = 1e-5
    priors = np.clip(priors, eps, 1 - eps)
    
    np.save(args.save_path, priors)
    print(f"Priors saved to {args.save_path}. Shape: {priors.shape}")
    
    # print the code snippet to copy to the model
    print("\n[!] Python snippet to load these priors in DAB-Deformable-DETR:")
    print("-" * 50)
    print(f"import numpy as np")
    print(f"import torch")
    print(f"from util.misc import inverse_sigmoid")
    print(f"priors = np.load('{os.path.abspath(args.save_path)}')")
    print(f"self.refpoint_embed.weight.data = inverse_sigmoid(torch.tensor(priors, dtype=torch.float32))")
    print("-" * 50)

if __name__ == '__main__':
    main()
