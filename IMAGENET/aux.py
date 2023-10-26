import os

from torchvision import datasets
import torchvision.models as torchmodels
import torchvision.transforms as transforms

from colossalai.utils import get_dataloader




def get_model(args, Num_classes):
	if args.pretrained:
		print("=> using pre-trained model '{}'".format(args.model))
		net = torchmodels.__dict__[args.model](pretrained=True)
	else:
		print("=> creating model '{}'".format(args.model), Num_classes)
		net = torchmodels.__dict__[args.model]()
	return net




def build_imagenet(argsM):
	argsM.data = '/ws/gim/IMAGENET/imagenet/'
	traindir = os.path.join(argsM.data, 'train')
	valdir = os.path.join(argsM.data, 'val')
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.229, 0.224, 0.225])

	train_dataset = datasets.ImageFolder(
		traindir,
		transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		]))

	val_dataset = datasets.ImageFolder(
		valdir,
		transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		]))
	train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=argsM.bsz, pin_memory=True, seed=1024*argsM.manualSeed, drop_last=True)
	test_dataloader = get_dataloader(dataset=val_dataset, batch_size=argsM.bsz, pin_memory=True, seed=1024*argsM.manualSeed, drop_last=True)
	relative_speeds = None
	return train_dataloader, test_dataloader, relative_speeds