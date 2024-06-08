import os
import torch
import shutil
import numpy as np
from torch import nn
from models.VisionTransformer.ViT import vit_base_patch16_224_in21k
from models.ResNet.RN import resnet50

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_model(args):
    if args.model == "ViT-B/16":
        model = vit_base_patch16_224_in21k(args.num_classes)
    elif args.model == "ResNet-50":
        model = resnet50(args.num_classes)
    else:
        raise Exception("Can't find any model name call {}".format(args.model))
    
    return model


def model_parallel(args, model):
    device_ids = [i for i in range(len(args.gpu.split(',')))]
    model = nn.DataParallel(model, device_ids=device_ids)

    return model

def remove_dir_and_create_dir(dir_name):
    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
        print(dir_name, "Creat OK")