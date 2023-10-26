import os

import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from colossalai.utils import get_dataloader


def get_model(args, out_size):
    if args.model   == 'resnet18':
        net = models.resnet18(pretrained=args.pretrained)
        net.fc = nn.Linear(in_features=512, out_features=out_size, bias=True)
    elif args.model == 'resnet34':
        net = models.resnet34(pretrained=args.pretrained)
        net.fc = nn.Linear(in_features=512, out_features=out_size, bias=True)
    elif args.model == 'resnet50':
        net = models.resnet50(pretrained=args.pretrained)
        net.fc = nn.Linear(in_features=2048, out_features=out_size, bias=True)
    elif args.model == 'resnet101':
        net = models.resnet50(pretrained=args.pretrained)
        net.fc = nn.Linear(in_features=2048, out_features=out_size, bias=True)
    elif args.model == 'resnet152':
        net = models.resnet50(pretrained=args.pretrained)
        net.fc = nn.Linear(in_features=2048, out_features=out_size, bias=True)
    else:
        print('==> Network not found...')
        exit()
    return net


def build_fg_loader(argsM):
    subpath = {'cub':'CUB_200_2011','cars':'Car196','fgvc':'fgvc_aricraft','dogs':'StanfordDogs'}
    argsM.datadir += subpath[argsM.dataset]

    traindir  = os.path.join(argsM.datadir, 'train')
    valdir    = os.path.join(argsM.datadir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
                    transforms.Resize(512),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.RandomCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ]))

    train_loader = get_dataloader(
        train_dataset, batch_size=argsM.bsz, shuffle=True, pin_memory=True, drop_last=True)

    val_loader = get_dataloader(
        datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(512),
                    transforms.CenterCrop(448),
                    transforms.ToTensor(),
                    normalize,
                ])),
        batch_size=argsM.bsz, shuffle=False,
        num_workers=0, pin_memory=True,drop_last=True)

    return train_loader, val_loader, None
