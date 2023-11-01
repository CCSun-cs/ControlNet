from share import *
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from imagenet_tutorial_dataset import MyDataset
# print("1")
from cldm.logger import ImageLogger
# print("2")
from cldm.model import create_model, load_state_dict
# print("3")
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import argparse
import time
from pytorch_lightning.callbacks import ModelCheckpoint

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # p.add_argument('--class-selector', type=str)
    # p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default='cifar10')
    # p.add_argument('--image-number', type=int, default=0)
    # p.add_argument('--save-name', type=str)
    # p.add_argument('--big-pic', action='store_true')
    p.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    p.add_argument('--seed', default=42, type=int)
    # nproc_per_node
    # p.add_argument('--nproc_per_node', default=1, type=int)
    # p.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # p.add_argument('--start_batch', default=0,type=int, required=True)
    # p.add_argument('--end_batch', default=100,type=int,required=True)
# Configs

     ###########################
    args = p.parse_args()
    # init_distributed_mode(args)
    # device = torch.device(args.device)
    seed = args.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # print("get_rank()###########",get_rank())
    # 获取可用的 GPU 数量
    available_gpus = torch.cuda.device_count()

    # 打印可用的 GPU 数量
    print(f"Available GPUs: {available_gpus}")

    # resume_path = './models/control_sd15_ini.ckpt'
    resume_path = '/egr/research-optml/sunchan5/ControlNet/checkpoints_imagenet_noprompt_new/epoch=3-loss=0.00-step=52199.ckpt'


    batch_size = 8
    logger_freq = 300######   每300个batch 保存一次数据
    # logger_freq = 1######
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.eval()

    # model.learning_rate = learning_rate
    # model.sd_locked = sd_locked
    # model.only_mid_control = only_mid_control


    # Misc
    # normal_loader = Data.DataLoader(normal_data, batch_size=batch_size, shuffle=False, worker_init_fn=lambda x: random.seed(seed))
    dataset = MyDataset()
    # dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,worker_init_fn=lambda x: random.seed(seed))

    logger = ImageLogger(batch_frequency=logger_freq)

    checkpoint_callback = ModelCheckpoint(
        # dirpath='./checkpoints/checkpoints_imagenet_prompt_new',  #############sccchange
            dirpath='./checkpoints/checkpoints_imagenet_noprompt_new',  #############sccchange
        filename='{epoch}-{loss:.2f}-{step:05d}',
        every_n_train_steps=logger_freq,
        save_top_k=3,  # 保留最佳的 3 个模型文件
        monitor= 'val/loss_simple_ema'
    )
    # 创建 PyTorch Lightning 的 Trainer
    trainer = pl.Trainer(
        gpus=available_gpus,  # 指定要使用的 GPU 数量
        precision=32,  # 设置精度
        callbacks=[logger,checkpoint_callback],  # 添加回调
        accelerator="ddp",  # 使用分布式数据并行 (DDP)
        num_nodes=1,  # 节点数，通常为 1
        # max_epochs=10,  # 训练的总周期数
        # resume_from_checkpoint='./checkpoints/your_checkpoint.ckpt',  # 设置从检查点文件恢复
    )
    print("Start training.")
    # Train!
    trainer.fit(model, dataloader)


# 在 PyTorch Lightning 中，trainer.fit(model, dataloader) 是如何执行模型的 forward 函数的详细流程如下：

# 数据加载器迭代：

# dataloader 是一个数据加载器对象，它包含了训练数据的批次。trainer 会迭代这些批次以进行训练。
# 批次数据传递到模型：

# trainer 从数据加载器中获取一个批次的数据。
# 批次数据（通常是输入数据）被传递给模型（model）的 forward 方法。
# 模型的 forward 方法执行：

# 在你的 model 中，你需要实现一个 forward 方法，该方法定义了模型的前向传播逻辑。这个方法接收输入数据，并计算模型的输出。
# forward 方法通常包括了模型中的各种层和操作，以处理输入数据，计算中间结果，并生成最终的输出。
# 模型输出返回：

# forward 方法的输出通常是模型的预测或中间结果。这个输出会被返回给 trainer。
# 损失计算和反向传播：

# trainer 接收模型的输出后，根据损失函数计算损失值。损失函数通常在 LightningModule 的 training_step 方法中定义。
# 然后，trainer 执行反向传播操作，计算损失相对于模型参数的梯度。
# 优化器更新：

# trainer 使用优化器来根据计算的梯度更新模型的参数。优化器通常在 LightningModule 的 optimizer_step 方法中定义。
# 重复迭代：

# 上述步骤会被重复执行，直到训练过程的指定轮数完成。
# 总结起来，trainer.fit(model, dataloader) 通过迭代数据加载器中的批次，
# 将数据传递给模型的 forward 方法，计算损失、执行反向传播和参数更新，以完成模型的训练。
# 你的模型的前向传播逻辑应该在 forward 方法中定义，PyTorch Lightning 会自动处理训练过程中的其他细节。