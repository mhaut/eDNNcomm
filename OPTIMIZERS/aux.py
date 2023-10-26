from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from colossalai.utils import get_dataloader

from models import *
from models.vit import ViT
from models.convmixer import ConvMixer



def get_model(args, Num_classes):
	if   args.model == 'vgg16':       net = VGGCIFAR('VGG16',   Num_classes=Num_classes)
	elif args.model == 'resnet18':    net = ResNet18(           Num_classes=Num_classes)
	elif args.model == 'resnet34':    net = ResNet34(           Num_classes=Num_classes)
	elif args.model == 'resnet50':    net = ResNet50(           Num_classes=Num_classes)
	elif args.model == 'resnet101':   net = ResNet101(          Num_classes=Num_classes)
	elif args.model == 'resnext29':   net = ResNeXt29_4x64d(    Num_classes=Num_classes)
	elif args.model == 'dla':         net = DLA(                Num_classes=Num_classes)
	elif args.model == 'densenet121': net = DenseNet121(        Num_classes=Num_classes)
	elif args.model == "convmixer":
		net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=Num_classes)
		args.lr = 1e-3
	elif args.model == "convmixerB":
		net = ConvMixer(256, 20, kernel_size=args.convkernel, patch_size=1, n_classes=Num_classes)
		args.lr = 1e-3
	elif args.model == "convmixer0":
		net = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=Num_classes)
	elif args.model == "convmixer1":
		net = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=Num_classes)
	elif args.model == "convmixer2":
		net = ConvMixer(1024, 20, kernel_size=9, patch_size=14, n_classes=Num_classes)
	elif args.model=="vit_small":
		from models.vit_small import ViT
		net = ViT(
		image_size = args.size,
		patch_size = args.patch,
		num_classes = Num_classes,
		dim = int(args.dimhead),
		depth = 6,
		heads = 8,
		mlp_dim = 512,
		dropout = 0.1,
		emb_dropout = 0.1
	)
		args.lr = 1e-4
	elif args.model=="vit":
		# ViT for cifar10
		net = ViT(
		image_size = args.size,
		patch_size = args.patch,
		num_classes = Num_classes,
		dim = int(args.dimhead),
		depth = 6,
		heads = 8,
		mlp_dim = 512,
		dropout = 0.1,
		emb_dropout = 0.1
	)
		args.lr = 1e-4
	elif args.model=="vit_timm":
		import timm
		net = timm.create_model("vit_base_patch16_384", pretrained=args.pretrained)
		net.head = nn.Linear(net.head.in_features, Num_classes)
		args.lr = 1e-4
	elif args.model=="vit_small_timm":
		import timm
		net = timm.create_model("vit_small_patch16_224", pretrained=args.pretrained)
		net.head = nn.Linear(net.head.in_features, Num_classes)
		args.lr = 1e-4
	elif args.model=="cait":
		from models.cait import CaiT
		net = CaiT(
		image_size = args.size,
		patch_size = args.patch,
		num_classes = Num_classes,
		dim = int(args.dimhead),
		depth = 6,   # depth of transformer for patch to patch attention only
		cls_depth=2, # depth of cross attention of CLS tokens to patch
		heads = 8,
		mlp_dim = 512,
		dropout = 0.1,
		emb_dropout = 0.1,
		layer_dropout = 0.05
	)
		args.lr = 1e-4
	elif args.model=="cait_small":
		from models.cait import CaiT
		net = CaiT(
		image_size = size,
		patch_size = args.patch,
		num_classes = Num_classes,
		dim = int(args.dimhead),
		depth = 6,   # depth of transformer for patch to patch attention only
		cls_depth=2, # depth of cross attention of CLS tokens to patch
		heads = 6,
		mlp_dim = 256,
		dropout = 0.1,
		emb_dropout = 0.1,
		layer_dropout = 0.05
	)
		args.lr = 1e-4
	elif args.model=="swin":
		from models.swin import swin_t
		net = swin_t(window_size=args.patch,
					num_classes=Num_classes,
					downscaling_factors=(2,2,2,1))
		args.lr = 1e-4
	return net



def build_cifar100(argsM):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	transform_test = transforms.Compose([
		transforms.Resize(size),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	train_dataset    = CIFAR100(root='/tmp/data', train=True, download=True, transform=transform_train)
	test_dataset     = CIFAR100(root='/tmp/data', train=False, download=True, transform=transform_test)
	train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=argsM.bsz, pin_memory=True, seed=1024*argsM.manualSeed)
	test_dataloader  = get_dataloader(dataset=test_dataset, batch_size=argsM.bsz, pin_memory=True, seed=1024*argsM.manualSeed)
	relative_speeds = None
	return train_dataloader, test_dataloader, relative_speeds



def build_cifar10(argsM):
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	transform_test = transforms.Compose([
		transforms.Resize(32),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	train_dataset    = CIFAR10(root='/tmp/data', train=True, download=True, transform=transform_train)
	test_dataset     = CIFAR10(root='/tmp/data', train=False, download=True, transform=transform_test)
	train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=argsM.bsz, pin_memory=True, seed=1024*argsM.manualSeed)
	test_dataloader  = get_dataloader(dataset=test_dataset, batch_size=argsM.bsz, pin_memory=True, seed=1024*argsM.manualSeed)
	relative_speeds = None
	return train_dataloader, test_dataloader, relative_speeds


def get_optim(net, args):
    if   args.optim == 'sgd':            optimizer = torch.optim.SGD(     net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optim == 'rmsprop':        optimizer = torch.optim.RMSprop( net.parameters(), lr=args.lr)
    elif args.optim == 'adam':           optimizer = torch.optim.Adam(    net.parameters(), lr=args.lr, weight_decay = args.wd)
    elif args.optim == 'adamw':          optimizer = torch.optim.AdamW(   net.parameters(), lr=args.lr, weight_decay = args.wd)
    elif args.optim == 'diffgrad':        optimizer = diffgrad(       net.parameters(), lr=args.lr)
    #elif optim_name == 'adabelief':      optimizer = AdaBelief(     net.parameters(), lr=args.lr)
    elif args.optim == 'adabelief':      optimizer = AdaBelief(net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decouple = args.weight_decouple, weight_decay = args.wd, fixed_decay = args.fixed_decay, rectify=False)
    elif args.optim == 'cosangulargrad': optimizer = cosangulargrad(net.parameters(), lr=args.lr)
    else:
        print('==> Optimizer not found...')
        exit()
    return optimizer
