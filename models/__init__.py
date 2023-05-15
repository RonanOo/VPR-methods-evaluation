import torch

from models import sfrs
from models import convap
from models import mixvpr
from models import netvlad
from models import gcl


def get_model(method, backbone=None, descriptors_dimension=None):
    if method == "sfrs":
        model = sfrs.SFRSModel()
    elif method == "netvlad":
        model = netvlad.NetVLAD(descriptors_dimension=descriptors_dimension)
    elif method == "cosplace":
        model = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=descriptors_dimension,
        )
    elif method == "mixvpr":
        model = mixvpr.get_mixvpr(descriptors_dimension=descriptors_dimension)
    elif method == "convap":
        model = convap.get_convap(descriptors_dimension=descriptors_dimension)
    elif method == "gcl":
        model = gcl.get_gcl(
            backbone=backbone,
            pool="GeM",
            norm="L2",
            mode="single",
            model_file="trained_models/gcl/gcl.pth.tar",
        )

    return model
