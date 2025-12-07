import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Return a ResNet-18 or ResNet-34 for CIFAR-10."""
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(weights=None)
    else:
        raise ValueError(f"Unknown model_name={model_name}")
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_cifar10_dataloaders(
    batch_size: int,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """Return train DataLoader (and optionally sampler) for CIFAR-10."""
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, sampler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device,
    max_steps: int | None = None,
) -> float:
    """Train for one epoch; return average loss."""
    model.train()
    running_loss = 0.0
    n_batches = 0

    for step, (images, targets) in enumerate(dataloader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)
