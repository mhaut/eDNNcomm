import os
import random
import time
import argparse

import torch

import sys
sys.path.append("/home/smoreno/ColossalAI/")
import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.engine.schedule import NonPipelineSchedule

import aux



# Define Replica
class Replica ():
	def __init__ (self, argsM = None):
		self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
		self.size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
		self.devices = ["gpu"] * int(self.size)
		self.device = torch.device('cuda' if (self.devices[self.rank].rstrip() == 'gpu' and torch.cuda.is_available()) else 'cpu')
		if self.device == torch.device('cuda'):
			self.rank_gpu = self.rank % torch.cuda.device_count()
			torch.cuda.set_device(self.rank_gpu)
			self.device = "cuda:" + str(self.rank_gpu)
		random.seed(100*argsM.manualSeed)
		torch.manual_seed(100*argsM.manualSeed)
		if self.device == torch.device('cuda'):
			torch.cuda.manual_seed_all(100)
		print("[",self.rank,"]  running in device:", self.device)


# Train
def main(argsM, p, epochs, chunks):
	disable_existing_loggers()
	logger = get_dist_logger()
	device = p.device
	if argsM.model == 'r101':
		argsM.bsz = 98 # Modification due to lack of space in GPU RAM.
	# build dataloader
	if argsM.dataset == "CIFAR10":
		train_dataloader, test_dataloader, speeds = aux.build_cifar10(argsM)
		classes = 10
	elif argsM.dataset == "CIFAR100":
		train_dataloader, test_dataloader, speeds = aux.build_cifar100(argsM)
		classes = 100
	else:
		print("DATASET NOT FOUND!")
		exit()
	model = aux.get_model(argsM, classes).to(device)
	criterion = torch.nn.CrossEntropyLoss()

	aux.get_optim()

	#############################################################################################
	# optionally resume from a checkpoint
	if argsM.resume:
		filename = '/tmp/checkpoint_' + str(int(os.environ['OMPI_COMM_WORLD_RANK'])) + '.pth.tar'
		if os.path.isfile(filename):
			print("=> loading checkpoint in device ", int(os.environ['OMPI_COMM_WORLD_RANK']))
			checkpoint = torch.load(filename)
			argsM.start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})".format(argsM.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(argsM.resume))
			exit()
	#############################################################################################
	else:
		argsM.start_epoch = 0
		
	lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=argsM.epochs)
	
	engine, _, _, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader, test_dataloader, lr_scheduler, sharing = argsM.sharingiter, algorithm = argsM.mode, speeds=speeds, balanced=argsM.balanced, device=device)
	timer = MultiTimer()
	schedule = NonPipelineSchedule()
	preconditioner = None
	trainer = Trainer(engine=engine, timer=timer, schedule=schedule, preconditioner=preconditioner, logger=logger, withkfac=argsM.withkfac, start_epoch=argsM.start_epoch)

	hook_list = [
		hooks.LossHook(),
		hooks.AccuracyHook(colossalai.nn.metric.Accuracy()),
		hooks.LogMetricByEpochHook(logger),
		hooks.LogTimingByEpochHook(timer,logger),
		hooks.LRSchedulerHook(lr_scheduler, by_epoch=True)
	]
	
	start = time.time()
	print("::", argsM.bsz)
	torch.cuda.empty_cache()
	trainer.fit(train_dataloader=train_dataloader,
				epochs=epochs,
				args=argsM,
				sharingiter=argsM.sharingiter,
				test_dataloader=test_dataloader,
				test_interval=1,
				hooks=hook_list,
				device=device,
				display_progress=False)
	
	final = time.time()-start
	print("Time: ", final)
	
	f = open("/tmp/times_"+argsM.model+"_"+argsM.mode+".txt", "a")
	f.write("%s\t%d\t%d\t%0.6f\t%0.6f\n" % (argsM.mode, argsM.sharingiter, argsM.epochs, final, final/argsM.epochs))
	f.close()


if __name__ == '__main__':
	parserM = argparse.ArgumentParser(description='PyTorch ColossalAI Training')
	parserM.add_argument('--balanced', default=1, type=int, help='balance')
	parserM.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
	parserM.add_argument('--manualSeed', default=1111, type=int, help='random seed')
	parserM.add_argument('--mode', default="1d", type=str, help='1d or proposed')
	parserM.add_argument('--lr', default=0.1, type=float)
	parserM.add_argument('--data', default="/p/project/prcoe12/imagenet/imagenet-1K-lmdb", type=str)
	parserM.add_argument('--bsz', default=128, type=int, help='batch size')
	parserM.add_argument('--epochs', default=400, type=int, help='number of epochs')
	parserM.add_argument('--chunks', default=1, type=int, help='pipeline chinks')
	parserM.add_argument('--speeds', default="0", type=str, help='processes speed')
	parserM.add_argument('--worldsize', default=1, type=int, help='number of processes')
	parserM.add_argument('--sharingiter', default=-1, type=int, help='global communication iters [automode = -1 (1 per epoch)]')
	parserM.add_argument('--withkfac', default='no', type=str, help='yes(True)/no(False)')
	parserM.add_argument('--model', default='r101', type=str, help='resnet model)')
	parserM.add_argument('--size', default="32", type=int)
	parserM.add_argument('--pretrained', default=False, type=bool)
	parserM.add_argument('--resume', default=False, type=bool)
	parserM.add_argument('--wd', default=0, type=float, metavar='W', help='weight decay (default: 1e-4)')
	parserM.add_argument('--beta1', default=0.9, type=float, help='beta1 in Adabelief')
	parserM.add_argument('--beta2', default=0.999, type=float, help='beta2 in Adabelief')
	argsM = parserM.parse_args()


	if argsM.sharingiter == -1:
		argsM.sharingiter = 50000 // (argsM.bsz * argsM.worldsize)
		if 50000 % (argsM.bsz * argsM.worldsize) != 0: argsM.sharingiter += 1

	p = Replica(argsM=argsM)
	CONFIG = dict(parallel=dict(
		data=dict(size=argsM.worldsize, mode=argsM.mode),
	))

	colossalai.launch(config=CONFIG,
			host=None,
			port=None,
			backend='mpi',
			rank = int(os.environ['OMPI_COMM_WORLD_RANK']),
			world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
			local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']),
			seed=argsM.manualSeed)

	main(argsM, p, argsM.epochs, argsM.chunks)
