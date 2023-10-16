from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms

from colossalai.utils import get_dataloader
from models import *




def get_model(args, Num_classes):
	if   args.model == 'vgg16':       net = VGGCIFAR('VGG16',   Num_classes=Num_classes)
	elif args.model == 'resnet18':    net = ResNet18(           Num_classes=Num_classes)
	elif args.model == 'resnet34':    net = ResNet34(           Num_classes=Num_classes)
	elif args.model == 'resnet50':    net = ResNet50(           Num_classes=Num_classes)
	elif args.model == 'resnet101':   net = ResNet101(          Num_classes=Num_classes)
	elif args.model == 'resnext29':   net = ResNeXt29_4x64d(    Num_classes=Num_classes)
	elif args.model == 'dla':         net = DLA(                Num_classes=Num_classes)
	elif args.model == 'densenet121': net = DenseNet121(        Num_classes=Num_classes)
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
