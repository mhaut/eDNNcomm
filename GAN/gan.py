import os
import random
import time
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

import sys
sys.path.append("/home/smoreno/ColossalAI/")
import colossalai
from colossalai.logging import get_dist_logger
from colossalai.utils import get_dataloader
from colossalai.trainer import hooks
from colossalai.nn.lr_scheduler import CosineAnnealingLR

import models
import aux


# Define Replica
class Replica ():
	def __init__ (self, manualSeed=100):
		self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
		self.size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
		self.devices = ["gpu"] * int(self.size)
		self.device = torch.device('cuda')
		if self.device == torch.device('cuda'):
			self.rank_gpu = self.rank % torch.cuda.device_count()
			torch.cuda.set_device(self.rank_gpu)
			self.device = "cuda:" + str(self.rank_gpu)
		random.seed(100 * manualSeed)
		torch.manual_seed(100 * manualSeed)
		if self.device == torch.device('cuda'):
			torch.cuda.manual_seed_all(100)
		print("[",self.rank,"]  running in device:", self.device)

os.makedirs("/tmp/images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', default=1111, type=int, help='random seed')
parser.add_argument('--worldsize', default=1, type=int, help='number of processes') 
parser.add_argument('--mode', default="1d", type=str, help='1d or proposed')
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--k", type=int, default=1, help="expansion factor")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()

p = Replica(manualSeed = opt.manualSeed)
device = p.device

CONFIG = dict(parallel=dict(
    data=dict(size=opt.worldsize, mode=opt.mode),
))

rank = int(os.environ['OMPI_COMM_WORLD_RANK'])


colossalai.launch(config=CONFIG,
        host=None,
        port=None,
        backend='mpi',
        rank = int(os.environ['OMPI_COMM_WORLD_RANK']),
        world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK']),
        seed=opt.manualSeed)


# build logger
logger = get_dist_logger()



# Initialize generator and discriminator
generator     = models.Generator(opt.n_classes, opt.latent_dim, opt.img_size, opt.channels, opt.k).to(device)
discriminator = models.Discriminator(opt.n_classes, opt.img_size, opt.channels, opt.k).to(device)

# Build criterion
criterion = torch.nn.CrossEntropyLoss().to(device)

adversarial_loss = torch.nn.BCELoss().to(device)
auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)

# Initialize weights
generator.apply(aux.weights_init_normal)
discriminator.apply(aux.weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

# lr_scheduler
lr_scheduler_G = CosineAnnealingLR(optimizer_G, total_steps=opt.n_epochs)
lr_scheduler_D = CosineAnnealingLR(optimizer_D, total_steps=opt.n_epochs)

# build datasets
train_dataset = MNIST(
    root='/tmp/data',
    train=True,
    download=False,
    transform=transforms.Compose(
        [
            transforms.Resize(opt.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
)

# build dataloaders
train_dataloader = get_dataloader(dataset=train_dataset, shuffle=True, batch_size=opt.batch_size)

def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim)))).to(device)
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels)).to(device)
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "/tmp/"+str(opt.mode)+"/%d.png" % batches_done, nrow=n_row, normalize=True)


engine_G, _, _, _ = colossalai.initialize(
    model=generator, optimizer=optimizer_G, criterion=criterion, lr_scheduler=lr_scheduler_G, algorithm=opt.mode
)
engine_D, _, _, _ = colossalai.initialize(
    model=discriminator, optimizer=optimizer_D, criterion=criterion, lr_scheduler=lr_scheduler_D, algorithm=opt.mode
)

hook_listG = [hooks.LRSchedulerHook(lr_scheduler_G, by_epoch=True)]
hook_listD = [hooks.LRSchedulerHook(lr_scheduler_D, by_epoch=True)]




# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    start = time.time()
    D_losses, G_losses = [], []
    for i, (imgs, labels) in enumerate(train_dataloader):
        if rank == 0:
            print("-------->", i, "de", len(train_dataloader))
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor)).to(device)
        labels = Variable(labels.type(LongTensor)).to(device)

        # -----------------
        #  Train Generator
        # -----------------

        # set gradients to zero
        engine_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim)))).to(device)
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size))).to(device)


        # Generate a batch of images (outputs)
        gen_imgs = engine_G(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = engine_D(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
        
        engine_G.backward(g_loss)
        engine_G.step(epoch=epoch)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        engine_D.zero_grad()
        
        # Loss for real images
        real_pred, real_aux = engine_D(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
        
        # Loss for fake images
        fake_pred, fake_aux = engine_D(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
        
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        engine_D.backward(d_loss)
        engine_D.step(epoch=epoch)

        batches_done = epoch * len(train_dataloader) + i
        if rank == 0:
            D_losses.append(d_loss)
            G_losses.append(g_loss)
    if rank == 0:
        f = open("/tmp/GAN_"+str(opt.mode)+".txt", "a")
        f.write('[%d/%d]: loss_d: %.3f, loss_g: %.3f\n' % (
            (epoch), opt.n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch), opt.n_epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))
    engine_G._call_sharingPIX(opt.mode, hook_listG)
    engine_D._call_sharingPIX(opt.mode, hook_listD)
