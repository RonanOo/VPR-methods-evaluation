import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import json
import os
import models
import parser
import commons
import visualizations
from test_dataset import TestDataset
from os.path import basename as bn
from pathlib import Path

import torchvision.transforms as ttf
from tqdm import tqdm
import sys
import torch
import os
import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np
import faiss

msls_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}

def extract_features(dl, net, f_length, feats_file):
    if not os.path.exists(feats_file):
        feats = np.zeros((len(dl.dataset), f_length))
        for i, batch in tqdm(enumerate(dl), desc="Extracting features"):
            if torch.cuda.is_available():
                x = net.forward(batch.cuda())
            else:
                x = net.forward(batch)
            feats[i * dl.batch_size:i * dl.batch_size + dl.batch_size] = x.cpu().detach().squeeze(0)

        np.save(feats_file, feats)
    else:
        print(feats_file, "already exists. Skipping.")

def extract_features_msls(subset, root_dir, net, f_length, savename, results_dir, batch_size, k, image_t=None, cls_token=False):
    cities = msls_cities[subset]

    result_file = results_dir + "/" + savename + "_predictions.txt"
    f = open(result_file, "w+")
    f.close()

    subset_dir = subset if subset == "test" else "train_val"
    for c in cities:
        print(c)
        m_raw_file = root_dir + subset_dir + "/" + c + "/database/raw.csv"
        q_idx_file = root_dir + subset_dir + "/" + c + "/query.json"
        m_idx_file = root_dir + subset_dir + "/" + c + "/database.json"
        
        test_ds = TestDataset(root_dir, q_idx_file)
        logging.info(f"Testing on {test_ds}")
        # Is shuffle supposed to be true?
        q_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=4, shuffle=True)
        
        q_feats_file = results_dir + "/" + savename + "_" + c + "_queryfeats.npy"
        if cls_token:
            q_feats_cls_file = results_dir + "/" + savename + "_" + c + "_queryfeats_cls.npy"
            extract_features(q_dl, net, f_length, q_feats_file, q_feats_cls_file)
        else:
            extract_features(q_dl, net, f_length, q_feats_file)
        m_ds = TestDataset(root_dir, m_idx_file)
        m_dl = DataLoader(m_ds, batch_size=batch_size, num_workers=4, shuffle=True)

        m_feats_file = results_dir + "/" + savename + "_" + c + "_mapfeats.npy"
        if cls_token:
            m_feats_cls_file = results_dir + "/" + savename + "_" + c + "_mapfeats_cls.npy"
            extract_features(m_dl, net, f_length, m_feats_file, m_feats_cls_file)
        else:
            extract_features(m_dl, net, f_length, m_feats_file)
        result_file = extract_msls_top_k(m_feats_file, q_feats_file, m_idx_file, q_idx_file, result_file, k, m_raw_file)
    if subset == "val":
        print(result_file)
        score_file = result_file.replace("_predictions", "_result")
        if not os.path.exists(score_file):
            validate(result_file, root_dir, score_file)

def extract_msls_top_k(map_feats_file, query_feats_file, db_idx_file, q_idx_file, result_file, k, m_raw_file=""):
    D, I = search(map_feats_file, query_feats_file, k)

    # load indices
    with open(db_idx_file, "r") as f:
        db_paths = np.array(json.load(f)["im_paths"])
    with open(q_idx_file, "r") as f:
        q_paths = np.array(json.load(f)["im_paths"])
    with open(result_file, "a+") as f:
        for i, q in enumerate(q_paths):
            q_id = q.split("/")[-1].split(".")[0]
            f.write(q_id + " " + " ".join([db_paths[j].split("/")[-1].split(".")[0] for j in I[i, :]]) + "\n")
    return result_file

def search(map_feats_file, query_feats_file, k=25):
    # load features
    query_feats = np.load(query_feats_file).astype('float32')
    map_feats = np.load(map_feats_file).astype('float32')
    if k is None:
        k = map_feats.shape[0]
    # build index and add map features
    index = faiss.IndexFlatL2(map_feats.shape[1])
    index.add(map_feats)
    # search top K
    D, I = index.search(query_feats.astype('float32'), k)
    return D, I


sys.path.append(os.environ['MAPILLARY_ROOT'])

from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.eval import eval

def validate(prediction, msls_root, result_file, ks=[1, 5, 10, 20]):
    # select for which ks to evaluate

    dataset = MSLS(msls_root, cities="", mode='val', posDistThr=25)

    # get query and positive image keys
    database_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages]
    positive_keys = [[','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.dbImages[pos]] for pos in dataset.pIdx]
    query_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.qImages[dataset.qIdx]]
    all_query_keys = [','.join([bn(i)[:-4] for i in p.split(',')]) for p in dataset.qImages]

    # load prediction rankings
    predictions = np.loadtxt(prediction, ndmin=2, dtype=str)

    # Ensure that there is a prediction for each query image
    for k in query_keys:
        assert k in predictions[:, 0], "You didn't provide any predictions for image {}".format(k)

    # Ensure that all predictions are in database
    for i, k in enumerate(predictions[:, 1:]):
        missing_elem_in_database = np.in1d(k, database_keys, invert = True)
        if missing_elem_in_database.all():

            print("Some of your predictions are not in the database for the selected task {}".format(k[missing_elem_in_database]))
            print("This is probably because they are panorama images. They will be ignored in evaluation")

            # move missing elements to the last positions of prediction
            predictions[i, 1:] = np.concatenate([k[np.invert(missing_elem_in_database)], k[missing_elem_in_database]])

    # Ensure that all predictions are unique
    for k in range(len(query_keys)):
        assert len(predictions[k, 1:]) == len(np.unique(predictions[k, 1:])), "You have duplicate predictions for image {} at line {}".format(query_keys[k], k)

    # Ensure that all query images are unique
    assert len(predictions[:,0]) == len(np.unique(predictions[:,0])), "You have duplicate query images"

    predictions = np.array([l for l in predictions if l[0] in query_keys])

    # evaluate ranks
    metrics = eval(query_keys, positive_keys, predictions, ks=ks)

    f = open(result_file, 'w') if result_file else None
    # save metrics
    for metric in ['recall', 'map']:
        for k in ks:
            line =  '{}_{}@{}: {:.3f}'.format("all",
                                              metric,
                                              k,
                                              metrics['{}@{}'.format(metric, k)])
            print(line)
            if f:
                f.write(line + '\n')
    if f:
        f.close()
    return metrics


args = parser.parse_arguments()
print(
    f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
)

model = models.get_model(
    args.method, args.backbone, args.descriptors_dimension, args.model_file
)
model = model.eval().to(args.device)

dataset = 'msls'
subset = 'val'
results_dir = "results/" + dataset + "/" + subset + "/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
savename = args.model_file.split("/")[-1].split(".")[0]
print(savename)
if "msls" in dataset.lower():
    extract_features_msls(subset, args.database_folder, model, args.descriptors_dimension, savename, results_dir, args.batch_size, 30)