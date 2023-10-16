import os
import random
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
    def __init__ (self, balance=1, cluster = None, argsM = None):
        self.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        self.size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        self.devices = ["gpu","gpu"] * int(self.size//2)
        self.device = torch.device('cuda' if (self.devices[self.rank].rstrip() == 'gpu' and torch.cuda.is_available()) else 'cpu')

        if self.device == torch.device('cuda'):
            self.rank_gpu = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(self.rank_gpu)
            self.device = "cuda:0" if self.rank_gpu == 0 else "cuda:1"

        random.seed(100*argsM.manualSeed)
        torch.manual_seed(100*argsM.manualSeed)
        if self.device == torch.device('cuda'):
            torch.cuda.manual_seed_all(100)

        print("[",self.rank,"]  running in device:", self.device)




# Train
def main(argsM, p, epochs):
    disable_existing_loggers()
    logger = get_dist_logger()

    # build model
    #if p.device == torch.device('cuda'):
    device = p.device

    train_dataloader, test_dataloader, speeds = aux.build_fg_loader(argsM)
    class_num={'cub':200,'cars':196,'fgvc':100,'dogs':120}
    model = aux.get_model(argsM, class_num[argsM.dataset])
    model = torch.nn.DataParallel(model).to(device)

    # build criterion
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), argsM.lr, momentum=argsM.momentum, weight_decay=argsM.weight_decay)
    lr_scheduler  = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)


    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion, train_dataloader, test_dataloader, lr_scheduler, sharing = argsM.sharing, algorithm = argsM.mode, speeds=speeds, balanced=argsM.balanced, device=device)
    timer = MultiTimer()
    schedule = NonPipelineSchedule()
    trainer = Trainer(engine=engine, timer=timer, schedule=schedule, logger=logger)

    hook_list = [
        hooks.LossHook(),
        hooks.AccuracyHook(col_nn.metric.Accuracy()),
        hooks.LogMetricByEpochHook(logger),
        hooks.LogTimingByEpochHook(timer,logger),
        hooks.LRSchedulerHook(lr_scheduler, by_epoch=True)
    ]

    start = time.time()
    trainer.fit(train_dataloader=train_dataloader,
                epochs=epochs,
                test_dataloader=test_dataloader,
                test_interval=1,
                hooks=hook_list,
                device=device,
                display_progress=False)

    print("Tiempo: ", time.time()-start)


if __name__ == '__main__':

    parserM = argparse.ArgumentParser(description='PyTorch ColossalAI Training')
    parserM.add_argument('--balanced', default=1, type=int, help='balance')
    parserM.add_argument('--dataset', default="cub", type=str, help='dataset')
    parserM.add_argument('--manualSeed', default=1111, type=int, help='random seed')
    parserM.add_argument('--mode', default="1d", type=str, help='1d or proposed')
    parserM.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parserM.add_argument('--speeds', default="0", type=str, help='processes speed')
    parserM.add_argument('--worldsize', default=1, type=int, help='number of processes')
    parserM.add_argument('--sharing', default=5, type=int, help='global communication epochs')
    parserM.add_argument('--model', default='r50p', type=str, help='resnet model)')
    parserM.add_argument('--pretrained', action='store_true')
    parserM.add_argument('--bsz', default=64, type=int, help='batch size')
    parserM.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parserM.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parserM.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parserM.add_argument('--datadir', type=str, help='path to dataset')

    argsM = parserM.parse_args()
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

    main(argsM, p, argsM.epochs)
