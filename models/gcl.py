from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F


def get_backbone(name):
    name = name.lower()
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
        output_dim = 2208
    elif name == "densenet121":
        backbone = models.densenet121(pretrained=True).features
        output_dim = 2208
    elif name == "vgg16":
        backbone = models.vgg16(pretrained=True).features
        output_dim = 512
    elif name == "resnext":
        backbone = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    if "resne" in name:
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        output_dim = 2048
    return backbone, output_dim


def get_gcl(
    name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese", model_file=""
):
    model = create_model(name=name, pool=pool, norm=norm, mode="single")
    try:
        model.load_state_dict(torch.load(model_file)["model_state_dict"])
    except:
        model.load_state_dict(torch.load(model_file)["state_dict"])
    model.eval()

    return model


def create_model(name, pool, last_layer=None, norm=None, p_gem=3, mode="siamese"):
    backbone, output_dim = get_backbone(name)
    layers = len(list(backbone.children()))

    if last_layer is None:
        last_layer = layers
    elif "densenet" in name:
        last_layer = last_layer * 2
    elif "vgg" in name:
        last_layer = last_layer * 8 - 2
    aux = 0
    for c in backbone.children():
        print(type(aux), " ", aux)
        print(type(layers), " ", layers)
        print(type(last_layer), " ", last_layer)
        if aux < layers - last_layer:
            print(aux, c._get_name(), "IS FROZEN")
            for p in c.parameters():
                p.requires_grad = False
        else:
            print(aux, c._get_name(), "IS TRAINED")
        aux += 1
    if mode == "siamese":
        return SiameseNet(backbone, pool, norm=norm, p=p_gem)
    else:
        return BaseNet(backbone, pool, norm=norm, p=p_gem)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class BaseNet(nn.Module):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(BaseNet, self).__init__()
        self.backbone = backbone
        for name, param in self.backbone.named_parameters():
            n = param.size()[0]
        self.feature_length = n
        if global_pool == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif global_pool == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        elif global_pool == "GeM":
            self.pool = GeM(p=p)
        else:
            self.pool = None
        self.norm = norm

    def forward(self, x0):
        out = self.backbone.forward(x0)
        out = self.pool.forward(out).squeeze(-1).squeeze(-1)
        if self.norm == "L2":
            out = nn.functional.normalize(out)
        return out


class SiameseNet(BaseNet):
    def __init__(self, backbone, global_pool=None, poolkernel=7, norm=None, p=3):
        super(SiameseNet, self).__init__(
            backbone, global_pool, poolkernel, norm=norm, p=p
        )

    def forward_single(self, x0):
        return super(SiameseNet, self).forward(x0)

    def forward(self, x0, x1):
        out0 = super(SiameseNet, self).forward(x0)
        out1 = super(SiameseNet, self).forward(x1)
        return out0, out1
