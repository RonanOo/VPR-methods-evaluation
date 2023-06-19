import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import models
import parser
import commons
import os
import visualizations
from test_dataset import TestDataset
import torchvision.transforms as ttf

from factory import extract_features_msls, extract_corrupt_features_msls, extract_features_map_query

corruptions = [
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "elastic_transform",
    "jpeg_compression",
    "rotate",
    "crop",
]

args = parser.parse_arguments()
print(
    f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
)

model = models.get_model(
    args.method, args.backbone, args.descriptors_dimension, args.model_file
)
model = model.eval().to(args.device)

dataset = "msls"

results_dir = "results/" + dataset + "/" + args.subset + "/" + args.method + "/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
savename = args.model_file.split("/")[-1].split(".")[0]
print(savename)
image_t = ttf.Compose([ttf.Resize(size=(480, 640)),
                               ttf.ToTensor(),
                               ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])

if dataset.lower() == "msls":
    if args.corruption is None:
        extract_features_msls(args.subset, args.root_dir, model, args.descriptors_dimension, image_t, savename, results_dir, args.batch_size, 30)
    else:
        assert args.corruption == "all" or args.corruption in corruptions, f"Choose a valid corruption: {corruptions}"
        if args.corruption == "all":
            for corruption in corruptions:
                print(f"Extracting features for {corruption} with all severities")
                for severity in range(1, 6):
                    print(f"Extracting features for {corruption} with severity {severity}")
                    extract_corrupt_features_msls(args.subset, args.root_dir, model, args.descriptors_dimension, image_t, savename, results_dir, args.batch_size, 30,
                                                    corruption, severity, False)
        elif args.severity:
            print(f"Extracting features for {args.corruption} with severity {args.severity}")
            extract_corrupt_features_msls(args.subset, args.root_dir, model, args.descriptors_dimension, image_t, savename, results_dir, args.batch_size, 30,
                                            args.corruption, int(args.severity), False)
        else:
            print(f"Extracting features for {args.corruption} with all severities")
            for severity in range(1, 6):
                print(f"Extracting features for {args.corruption} with severity {severity}")
                extract_corrupt_features_msls(args.subset, args.root_dir, model, args.descriptors_dimension, image_t, savename, results_dir, args.batch_size, 30,
                                                args.corruption, severity, False)
else:
    extract_features_map_query(args.root_dir, args.query_idx_file, args.map_idx_file, model, args.descriptors_dimension, savename, results_dir, args.batch_size, 30, dataset.lower())