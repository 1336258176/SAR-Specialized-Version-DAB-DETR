import json
import numpy as np
import argparse
import torch
import logging
import os
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser("K-Means for DAB-Deformable-DETR prior boxes")
    parser.add_argument('--anno_file', '-i', type=str, required=True, help='Path to COCO instances_train.json')
    parser.add_argument('--num_clusters', type=int, default=300, help='Number of prior boxes (queries)')
    parser.add_argument('--output', '-o', type=str, default='tools/priors', help='Save path for priors directory')
    return parser.parse_args()

def get_kmeans_anchors(coco_json_path, output_path, num_clusters=300):
    logging.info(f"正在读取COCO标注文件: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # 建立 image_id 到 image 尺寸的映射，用于归一化
    img_id_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}

    wh_list = []
    logging.info("正在提取并归一化真实边界框宽高...")
    for ann in data['annotations']:
        img_w, img_h = img_id_to_size[ann['image_id']]
        # COCO bbox 格式: [x_min, y_min, width, height]
        _, _, w, h = ann['bbox']
        
        norm_w = w / img_w
        norm_h = h / img_h
        
        # 过滤掉异常的极小框
        if norm_w > 0.001 and norm_h > 0.001:
            wh_list.append([norm_w, norm_h])

    wh_array = np.array(wh_list)
    logging.info(f"共提取有效边界框数量: {len(wh_array)}")

    logging.info(f"正在进行 K-Means 聚类 (K={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(wh_array)
    
    cluster_centers = kmeans.cluster_centers_
    prior_wh = torch.tensor(cluster_centers, dtype=torch.float32)
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "hrsid_prior_wh.pt")
    
    torch.save(prior_wh, output_file)
    
    logging.info("聚类完成！")
    logging.info(f"前5个聚类中心 (归一化 w, h):\n{prior_wh[:5]}")
    logging.info(f"已将先验宽高保存至: {output_file}")


if __name__ == '__main__':
    args = parse_args()
    get_kmeans_anchors(args.anno_file, args.output, args.num_clusters)