import torch
import sys
import torch.nn as nn
import os.path as osp
import torchvision.models as models
import torch.nn.functional as F
from . import (
    conv3,
    lenet,
    wresnet,
    resnet,
    Alexnet,
    conv3_gen,
    conv3_cgen,
    conv3_dis,
    conv3_mnist,
    Densenet,
    Mobilenet,
    vit,
    vit_small,
    swin,
    levit,
)
from .cifar10_models import resnet18, resnet34,resnet50, vgg13,vgg16,vgg13_bn,vgg16_bn,vgg19_bn, vgg19
from .Text_Classfication import dpcnn, vdcnn_9, vdcnn_17, vdcnn_29, vdcnn_49, textcnn, textrcnn,fasttext,rnnattention,transformer,han,rnn
from datasets import get_nclasses


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model_dict = {
    "conv3_gen": conv3_gen.conv3_gen,
    "conv3_cgen": conv3_cgen.conv3_cgen,
    "conv3_dis": conv3_dis.conv3_dis,
    "lenet": lenet.lenet,
    "conv3": conv3.conv3,
    "conv3_mnist": conv3_mnist.conv3_mnist,
    "wres22": wresnet.WideResNet,
    "wres22_4": wresnet.WideResNet,
    "wres22_8": wresnet.WideResNet,
    "wres28": wresnet.WideResNet,
    "wres34": wresnet.WideResNet,
    "wres40": wresnet.WideResNet,
    "wres52": wresnet.WideResNet,
    "res20": resnet.resnet20,
    "res20_1": resnet.resnet20_1,
    "res20_2": resnet.resnet20_2,
    "res32": resnet.resnet32,
    "res32_1": resnet.resnet32_1,
    "res32_2": resnet.resnet32_2,
    "res44": resnet.resnet44,
    "res44_1": resnet.resnet44_1,
    "res44_2": resnet.resnet44_2,
    "res56": resnet.resnet56,
    "res56_1": resnet.resnet56_1,
    "res56_2": resnet.resnet56_2,
    "res62": resnet.resnet62,
    "res62_1": resnet.resnet62_1,
    "res62_2": resnet.resnet62_2,
    "res86": resnet.resnet86,
    "res86_1": resnet.resnet86_1,
    "res86_2": resnet.resnet86_2,
    "res98": resnet.resnet98,
    "res98_1": resnet.resnet98_1,
    "res98_2": resnet.resnet98_2,
    "res110": resnet.resnet110,
    "res110_1": resnet.resnet110_1,
    "res110_2": resnet.resnet110_2,
    "res18": resnet18, 
    "res34": resnet34,
    "res50": resnet50,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
    "vgg13": vgg13,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "Alexnet": Alexnet.Alexnet,
    "Densenet121":Densenet.DenseNet121,
    "Densenet169":Densenet.DenseNet169,
    "Densenet201":Densenet.DenseNet201,
    "Mobilenet":Mobilenet.MobileNet,
    "vit":vit.vit,
    "vit_small":vit_small.vit_small,
    "vit_tiny":vit_small.vit_tiny,
    "swin_T":swin.swin_T,
    "swin_S":swin.swin_S,
    "swin_B":swin.swin_B,
    "swin_L":swin.swin_L,
    "levit_128":levit.levit_128,
    "levit_128s":levit.levit_128s,
    "levit_192":levit.levit_192,
    "levit_256":levit.levit_256,
    "levit_384":levit.levit_384,

    "dpcnn":dpcnn,
    "vdcnn_9":vdcnn_9,
    "vdcnn_17":vdcnn_17,
    "vdcnn_29":vdcnn_29,
    "vdcnn_49":vdcnn_49,
    "textcnn":textcnn,
    "textrcnn":textrcnn,
    "fasttext":fasttext,
    "rnnattention":rnnattention,
    "transformer":transformer,
    "han":han,
    "rnn":rnn,
}

gen_channels_dict = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "fashionmnist": 1,

}

gen_dim_dict = {
    "cifar10": 8,
    "cifar100": 8,
    "gtsrb": 8,
    "svhn": 8,
    "mnist": 7,
    "fashionmnist": 7,
    "ag_news": 4,
    "celeba":8,
}

in_channel_dict = {
    "cifar10": 3,
    "celeba":3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "mnist": 1,
    "fashionmnist": 1,
}


def get_model(modelname, dataset="", pretrained=None, latent_dim=10, **kwargs):
    model_fn = model_dict[modelname]
    num_classes = get_nclasses(dataset)

    if modelname in [
        "conv3",
        #"lenet",
        "res20",
        "res20_1",
        "res20_2",
        "res32",
        "res32_1",
        "res32_2",
        "res44",
        "res44_1",
        "res44_2",
        "res56",
        "res56_1",
        "res56_2",
        "res62",
        "res62_1",
        "res62_2",
        "res86",
        "res86_1",
        "res86_2",
        "res98",
        "res98_1",
        "res98_2",
        "res110",
        "res110_1",
        "res110_2",
        "conv3_mnist",

    ]:
        model = model_fn(num_classes)
    
    elif modelname in ["lenet","Alexnet","Densenet121","Densenet169","Densenet201","Mobilenet"]:
        if dataset in ["mnist","fashionmnist"]:
            model= model_fn(
                in_channels=1,
                num_classes=num_classes
            )
        else:
            model= model_fn(
                in_channels=3,
                num_classes=num_classes,
            )
            
    elif modelname in ["dpcnn","vdcnn_9","vdcnn_17","vdcnn_29","vdcnn_49","textcnn","textrcnn","fasttext","rnnattention","transformer","han","rnn"]:
        if dataset in ["ag_news"]:
            model= model_fn()

    elif modelname in ["vit","vit_small","vit_tiny","swin_T","swin_S","swin_B","swin_L","levit_128","levit_128s","levit_192","levit_256","levit_384",]:
        #if dataset in ["ag_news"]:
        model= model_fn()
        
    elif modelname == "wres22":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=22,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=22, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
        print("num_classes",num_classes)

    elif modelname == "wres22_4":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=22,
                num_classes=num_classes,
                widen_factor=4,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=22, num_classes=num_classes, widen_factor=4, dropRate=0.0
            )
    elif modelname == "wres22_8":
            if dataset in ["mnist", "fashionmnist"]:
                model = model_fn(
                    depth=22,
                    num_classes=num_classes,
                    widen_factor=8,
                    dropRate=0.0,
                    upsample=True,
                    in_channels=1,
                )
            else:
                model = model_fn(
                    depth=22, num_classes=num_classes, widen_factor=8, dropRate=0.0
                )

    elif modelname == "wres34":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=34,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=34, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname == "wres28":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=28,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=28, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname == "wres40":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=40,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname == "wres52":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=52,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=52, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname in ["conv3_gen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
        )
    elif modelname in ["conv3_cgen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
            n_classes=num_classes,
        )
    elif modelname in ["conv3_dis"]:
        model = model_fn(channels=gen_channels_dict[dataset], dataset=dataset)
    elif modelname in ["res18", "res34", "res50", "vgg13_bn","vgg16_bn", "vgg19_bn", "vgg19", "vgg13", "vgg16"]:
        model = model_fn(pretrained=pretrained,
                         num_classes=num_classes,)
    else:
        sys.exit("unknown model")

    return model