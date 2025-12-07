import os
import time
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, distributed

from train_utils import get_model, get_cifar10_dataloaders, train_one_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-gpu-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps-per-epoch", type=int, default=-1)
    parser.add_argument("--output-file", type=str, default="ddp_results.json")
    args = parser.parse_args()

    max_steps = args.max_steps_per_epoch if args.max_steps_per_epoch > 0 else None

    # Environment variables set by torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(backend="nccl")

    model = get_model(args.model_name).to(device)
    model = DDP(model, device_ids=[local_rank])

    train_loader, sampler = get_cifar10_dataloaders(
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        distributed_flag=True,
        rank=rank,
        world_size=world_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    epoch_times = []

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        torch.cuda.synchronize()
        start = time.time()

        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_steps=max_steps,
        )

        torch.cuda.synchronize()
        end = time.time()
        epoch_time = end - start

        if rank == 0:
            epoch_times.append(epoch_time)
            print(f"[DDP] Epoch {epoch+1}/{args.epochs} - "
                  f"loss: {avg_loss:.4f} - time: {epoch_time:.3f} s",
                  flush=True)

    # Only rank 0 writes the results file
    if rank == 0:
        with open(args.output_file, "w") as f:
            json.dump({"epoch_times": epoch_times}, f)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
