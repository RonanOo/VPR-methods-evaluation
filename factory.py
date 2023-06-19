import torchvision.transforms as ttf
from tqdm import tqdm
import sys
import torch
import os
import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np
import faiss
from torch.utils.data import DataLoader
from torchvision import models
from torch.utils.data import Dataset
import json
import h5py
from scipy.spatial.distance import squareform
from PIL import Image
import math
import pandas as pd
from scipy.spatial.distance import cdist

from os.path import basename as bn
from pathlib import Path

sys.path.append(os.environ['MAPILLARY_ROOT'])

from mapillary_sls.datasets.msls import MSLS
from mapillary_sls.utils.eval import eval

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

default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'toy': ["amman"],
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

def extract_features_msls(subset, root_dir, net, f_length, image_t, savename, results_dir, batch_size, k, cls_token=False):
    cities = default_cities[subset]

    result_file = results_dir + "/" + savename + "_predictions.txt"
    f = open(result_file, "w+")
    f.close()

    subset_dir = subset if subset == "test" else "test"
    for c in cities:
        print(c)
        m_raw_file = root_dir + subset_dir + "/" + c + "/database/raw.csv"
        q_idx_file = root_dir + subset_dir + "/" + c + "/query.json"
        m_idx_file = root_dir + subset_dir + "/" + c + "/database.json"
        q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
        q_feats_file = results_dir + "/" + savename + "_" + c + "_queryfeats.npy"
        if cls_token:
            q_feats_cls_file = results_dir + "/" + savename + "_" + c + "_queryfeats_cls.npy"
            extract_features(q_dl, net, f_length, q_feats_file, q_feats_cls_file)
        else:
            extract_features(q_dl, net, f_length, q_feats_file)
        m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
        m_feats_file = results_dir + "/" + savename + "_" + c + "_mapfeats.npy"
        if cls_token:
            m_feats_cls_file = results_dir + "/" + savename + "_" + c + "_mapfeats_cls.npy"
            extract_features(m_dl, net, f_length, m_feats_file, m_feats_cls_file)
        else:
            extract_features(m_dl, net, f_length, m_feats_file)
        dists_file = distances(q_feats_file, m_feats_file)
        result_file = extract_msls_top_k(dists_file, m_idx_file, q_idx_file, result_file, k, m_raw_file)
    if subset == "val":
        print(result_file)
        score_file = result_file.replace("_predictions", "_result")
        if not os.path.exists(score_file):
            validate(result_file, root_dir, score_file)

def extract_corrupt_features_msls(subset, root_dir, net, f_length, image_t, savename, results_dir, batch_size, k, corruption, severity, saveImages):
    cities = default_cities[subset]

    result_file = results_dir + "/" + savename + "_" + corruption + "_" + str(severity) + "_predictions.txt"
    f = open(result_file, "w+")
    f.close()
    for c in cities:
        print(c)
        m_raw_file = root_dir + "test/" + c + "/database/raw.csv"
        q_idx_file = root_dir + "test/" + c + "/query.json"
        m_idx_file = root_dir + "test/" + c + "/database.json"
        q_dl = create_dataloader("test_corrupt", root_dir, q_idx_file, None, image_t, batch_size, corruption, severity, saveImages)
        q_feats_file = results_dir + "/" + savename + "_" + c + "_" + corruption + "_" + str(severity) + "_queryfeats.npy"
        extract_features(q_dl, net, f_length, q_feats_file)
        m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
        m_feats_file = results_dir + "/" + savename + "_" + c + "_mapfeats.npy"
        extract_features(m_dl, net, f_length, m_feats_file)
        dists_file = distances(q_feats_file, m_feats_file)
        extract_msls_top_k(dists_file, m_idx_file, q_idx_file, result_file, k, m_raw_file)
    if subset == "val":
        metrics = validate(result_file, root_dir, result_file.replace("predictions", "result"))
        print(metrics)


def extract_features_map_query(root_dir, q_idx_file, m_idx_file, net, f_length,savename, results_dir,batch_size, k, ds):
    q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
    q_feats_file =results_dir+"/"+savename+"_queryfeats.npy"
    extract_features(q_dl, net, f_length, q_feats_file)
    m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
    m_feats_file =results_dir+"/"+savename+"_mapfeats.npy"
    extract_features(m_dl, net, f_length, m_feats_file)
    dists_file=distances(q_feats_file,m_feats_file)
    result_file=results_dir+"/"+savename+"_predictions.npy"
    extract_top_k(dists_file, result_file, k)
    if ds == "robotcarseasons":
        predict_poses(root_dir, dists_file)
    elif ds == "extendedcmu":
        predict_poses_cmu(root_dir, dists_file)
    elif ds == "pitts30k" or ds == "tokyo247":
        eval_pitts(root_dir, ds, result_file)

def distances(query_feats_file, map_feats_file):
    dists_file=query_feats_file.replace("_queryfeats.npy", "_distances.npy")
    if not os.path.exists(dists_file):
        query_feats=np.load(query_feats_file)
        map_feats=np.load(map_feats_file)
        n = len(query_feats)
        m = len(map_feats)
        dists = np.zeros(( n,m), dtype="float16")
        aux = 0
        for i in tqdm(range(m), desc="Calculating distances"):
            dists[:,i] = cdist(map_feats[i:i + 1, :], query_feats).flatten().astype("float16")
            aux += n - 1 - i
        dists= dists.astype("float16")
        np.save(dists_file, dists)
    else:
        print(dists_file,"already exists. Skipping.")
    return dists_file

def load_index(index):
    with open(index) as f:
        data = json.load(f)
    im_paths = np.array(data["im_paths"])
    im_prefix = data["im_prefix"]

    if "poses" in data.keys():
        poses = np.array(data["poses"])
        return im_paths, poses, im_prefix
    else:
        return im_paths, im_prefix


def world_to_camera(pose):
    [w_qw, w_qx, w_qy, w_qz, w_tx, w_ty, w_tz] = pose
    r = R.from_quat([w_qx, w_qy, w_qz, w_qw]).as_matrix().T
    tx, ty, tz = np.dot(np.array([w_tx, w_ty, w_tz]), np.linalg.inv(-r))
    qx, qy, qz, qw = R.from_matrix(r).as_quat()
    return qw, qx, qy, qz, tx, ty, tz

def predict_poses_cmu(root_dir, m_feats_file, q_feats_file):
    ref_impaths, ref_poses, ref_impref = load_index(root_dir + "reference.json")
    test_impaths, test_impref = load_index(root_dir + "test.json")
    name = "ExtendedCMU" if "extended" in m_feats_file else "CMU"
    D, I = search(m_feats_file, q_feats_file, 1)
    name = m_feats_file.replace("_mapfeats", "_toeval").replace("/MSLS_", "/" + name + "_eval_MSLS_").replace(".npy",
                                                                                                              ".txt")
    with open(name, "w") as f:
        for q, db_index in tqdm(zip(test_impaths, I), desc="Predicting poses..."):
            cut_place = q.find("/img")
            q_im_tosubmit = q[cut_place + 1:]
            pose = np.array((ref_poses[db_index])).flatten()
            submission = q_im_tosubmit + " " + " ".join(pose.astype(str)) + "\n"
            f.write(submission)


def predict_poses(root_dir, m_feats_file, q_feats_file):
    ref_impaths, ref_poses, ref_impref = load_index(root_dir + "reference.json")
    test_impaths, test_impref = load_index(root_dir + "test.json")

    D, best_score = search(m_feats_file, q_feats_file, 1)

    name = m_feats_file.replace("_mapfeats", "_toeval").replace("/MSLS_", "/RobotCar_eval_MSLS_").replace(".npy",
                                                                                                          ".txt")
    with open(name, "w") as f:
        for q_im, db_index in tqdm(zip(test_impaths, best_score), desc="Predicting poses..."):
            cut_place = q_im.find("/rear")
            q_im_tosubmit = q_im[cut_place + 1:]
            assert q_im_tosubmit.startswith("rear/")
            pose = np.array(world_to_camera(ref_poses[db_index].flatten()))
            submission = q_im_tosubmit + " " + " ".join(pose.astype(str)) + "\n"
            f.write(submission)


def eval_pitts(root_dir, ds, result_file):
    if "pitts" in ds:
        gt_file = root_dir + ds + "_test_gt.h5"
    elif ds.lower() == "tokyotm":
        gt_file = root_dir + "val_gt.h5"
    else:
        gt_file = root_dir + "gt.h5"
    ret_idx = np.load(result_file)
    score_file = result_file.replace("predictions.npy", "scores.txt")
    print(ret_idx.shape)
    ks = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    with open(score_file, "w") as sf:
        with h5py.File(gt_file, "r") as f:
            gt = f["sim"]
            print(gt.shape)
            for k in ks:
                hits = 0
                total = 0
                for q_idx, ret in enumerate(ret_idx):
                    if np.any(gt[q_idx, :]):
                        total += 1
                        db_idx = sorted(ret[:k])
                        hits += np.any(gt[q_idx, db_idx])
                print(k, np.round(hits / total * 100, 2))
                sf.write(str(k) + "," + str(np.round(hits / total * 100, 2)) + "\n")


def extract_features_map_query(root_dir, q_idx_file, m_idx_file, net, f_length, savename, results_dir, batch_size, k,
                               ds):
    q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
    q_feats_file = results_dir + "/" + savename + "_queryfeats.npy"
    extract_features(q_dl, net, f_length, q_feats_file)
    m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
    m_feats_file = results_dir + "/" + savename + "_mapfeats.npy"
    extract_features(m_dl, net, f_length, m_feats_file)
    result_file = results_dir + "/" + savename + "_predictions.npy"
    if ds.lower() == "tokyotm":
        extract_top_k_tokyotm(m_feats_file, q_feats_file, m_idx_file, q_idx_file, result_file, k)
    else:
        extract_top_k(m_feats_file, q_feats_file, result_file, k)
    if ds == "robotcarseasons":
        predict_poses(root_dir, m_feats_file, q_feats_file)
    elif ds == "extendedcmu" or ds == "cmu":
        predict_poses_cmu(root_dir, m_feats_file, q_feats_file)
    elif "pitts" in ds or "tokyo" in ds:
        eval_pitts(root_dir, ds, result_file)


def extract_top_k_tokyotm(m_feats_file, q_feats_file, db_idx_file, q_idx_file, result_idx_file, k):
    print("TokyoTM")
    D, best_score = search(m_feats_file, q_feats_file)
    with open(db_idx_file, "r") as f:
        db_paths = np.array(json.load(f)["im_paths"])
    with open(q_idx_file, "r") as f:
        q_paths = np.array(json.load(f)["im_paths"])
    result_idx = np.zeros((len(q_paths), k))
    for i, q in enumerate(q_paths):
        q_timestamp = int(q.split("/")[3][1:])
        aux = 0
        for t in range(k):
            idx = best_score[i, aux]
            db = db_paths[idx]
            db_timestamp = int(db.split("/")[3][1:])

            while (np.abs(q_timestamp - db_timestamp) < 1):  # ensure we retrieve something at least a month away
                aux += 1
                idx = best_score[i, aux]
                db = db_paths[idx]
                db_timestamp = int(db.split("/")[3][1:])
            result_idx[i, t] = best_score[i, aux]
            aux += 1

    np.save(result_idx_file, result_idx.astype(int))


def extract_msls_top_k(dists_file, db_idx_file, q_idx_file, result_file, k,m_raw_file=""):
    dists=np.load(dists_file)
    with open(db_idx_file, "r") as f:
        db_paths=np.array(json.load(f)["im_paths"])
    with open(q_idx_file, "r") as f:
        q_paths=np.array(json.load(f)["im_paths"])
    best_score = np.argsort(dists, axis=1)
    result_idx=np.zeros((len(q_paths),k+1))
    with open(result_file, "a+") as f:
        for i,q in enumerate(q_paths):
            result_idx[i,0]=i 
            result_idx[i,1:]=best_score[i,:k]
            q_id=q.split("/")[-1].split(".")[0]
            f.write(q_id+" "+" ".join([db_paths[j].split("/")[-1].split(".")[0] for j in best_score[i,:k]])+"\n")
    np.save(result_file, result_idx)
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


def extract_top_k(map_feats_file, query_feats_file, result_file, k):
    D, I = search(map_feats_file, query_feats_file, k)
    np.save(result_file, I)
    
def create_dataloader(dataset, root_dir, idx_file, gt_file, image_t, batch_size, corruption=None, severity=None, saveImages=False):
    # Create dataset
    if dataset=="test":
        ds = TestDataSet(root_dir, idx_file, transform=image_t)
        return DataLoader(ds, batch_size=batch_size, num_workers=4)
    if dataset=="test_corrupt":
        ds = CorruptDataset(root_dir, idx_file, corruption=corruption, severity=severity, saveImages=saveImages, transform=image_t)
        return DataLoader(ds, batch_size=batch_size, num_workers=4)

    if dataset == "soft_siamese":
        ds = SiameseDataSet(root_dir, idx_file, gt_file, ds_key="fov", transform=image_t)
    elif dataset == "binary_siamese":
        ds = SiameseDataSet(root_dir, idx_file, gt_file, ds_key="sim", transform=image_t)
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)
    


def create_msls_dataloader(dataset, root_dir, cities, transform, batch_size,model=None):
    if dataset == "binary_MSLS":
        ds = MSLSDataSet(root_dir, cities, ds_key="sim", transform=transform)
    elif dataset == "soft_MSLS":
        ds = MSLSDataSet(root_dir, cities, ds_key="fov", transform=transform)
    return DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=True)


def get_backbone(name):
    if name == "resnet18":
        backbone = models.resnet18(pretrained=True)
    elif name == "resnet34":
        backbone = models.resnet34(pretrained=True)
    elif name == "resnet152":
        backbone = models.resnet152(pretrained=True)
    elif name == "resnet50":
        backbone = models.resnet50(pretrained=True)
    if name == "densenet161":
        backbone = models.densenet161(pretrained=True).features
        output_dim=2208
    elif name == "densenet121":
        backbone = models.densenet121(pretrained=True).features
        output_dim=2208
    elif name == "vgg16":
        backbone = models.vgg16(pretrained=True).features
        output_dim = 512
    elif name == "resnext":
        backbone = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    if "resne" in name:
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        output_dim = 2048
    return backbone, output_dim


def create_model(name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese"):
    backbone, output_dim = get_backbone(name)
    layers = len(list(backbone.children()))

    if last_layer is None:
        last_layer = layers
    elif "densenet" in name:
        last_layer=last_layer*2
    elif "vgg" in name:
        last_layer=last_layer*8-2
    aux = 0
    for c in backbone.children():

        if aux < layers - last_layer:
            print(aux, c._get_name(), "IS FROZEN")
            for p in c.parameters():
                p.requires_grad = False
        else:
            print(aux, c._get_name(), "IS TRAINED")
        aux += 1
    if mode=="siamese":
        return SiameseNet(backbone, pool, norm=norm, p=p_gem)
    else:
        return BaseNet(backbone, pool, norm=norm, p=p_gem)


class BaseDataSet(Dataset):
    def __init__(self, root_dir, idx_file, gt_file=None, ds_key="sim", transform=None):
        """
        Args:
            idx_file (string): Path to the idx file (.json)
            gt_file (string): Path to the GT file with pairwise similarity (.h5).
            ds_key (string): dataset name
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_paths = self.load_idx(idx_file)
        self.root_dir = root_dir
        self.ds_key=ds_key
        if gt_file is not None:
            with h5py.File(gt_file, "r") as f:
                self.gt_matrix = torch.Tensor((f[ds_key][:].flatten()).astype(float))
        self.transform = transform
        self.n = len(self.im_paths)

    @staticmethod
    def load_idx(idx_file):
        with open(idx_file) as f:
            data = json.load(f)
            root_dir = data["im_prefix"]
            im_paths = data["im_paths"]
            return im_paths

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        pass

    def read_image(self, impath):
        img_name = os.path.join(self.root_dir,
                                impath)
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image

class TestDataSet(BaseDataSet):
    def __init__(self, root_dir, idx_file, transform=None):
        super().__init__(root_dir, idx_file, None, transform=transform)

    def __getitem__(self, idx_im):
        return self.read_image(self.im_paths[idx_im])

class CorruptDataset(TestDataSet):
    def __init__(self, root_dir, idx_file, corruption, severity=1, saveImages=False, transform=None):
        super().__init__(root_dir, idx_file, transform)
        self.saveImages = saveImages
        self.corruption = corruption
        self.severity = severity

    def read_image(self, impath):
        corrpath = impath.replace(".jpg", f"_{self.corruption}_{self.severity}.jpg")

        if self.saveImages:
            # Open the array and convert it to a NumPy array
            imageArrays = np.array(Image.open(os.path.join(self.root_dir, impath)))
            # Initialise the RandomState
            hashString = impath + self.corruption + str(self.severity)
            randState = RandomState(bytearray(hashString.encode()))
            # Corrupt the image
            corruptedArray = corrupt(imageArrays, randState, severity=self.severity, corruption_name=self.corruption)
            # Convert image back to PIL object
            corruptedImage = Image.fromarray(corruptedArray)

            # Save image for next run
            corrupted_name = os.path.join(self.root_dir, corrpath)
            corruptedImage.save(corrupted_name)

        return super().read_image(corrpath)
    
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



