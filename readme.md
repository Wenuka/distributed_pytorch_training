# Multi-GPU Training Benchmark: Single GPU vs DataParallel vs DistributedDataParallel

This repository benchmarks the performance of Single-GPU training, PyTorch DataParallel (DP), and DistributedDataParallel (DDP) using ResNet models on CIFAR-10.
Training was executed on *vast.ai* using two consumer GPUs.

---

## Environment Setup

The following software and hardware versions were used:

```
Python 3.12.11
PyTorch: 2.8.0
CUDA version (compiled): 12.8
cuDNN version: 91002
Num GPUs: 2
- GPU 0: NVIDIA GeForce RTX 4070 Ti
- GPU 1: NVIDIA GeForce RTX 4070 Ti
```

---

## How to Run

use the `multiGPU_training.ipynb` to run the experiment.

To replicate this experiment, ensure the following files are in the same directory:
*   `multiGPU_training.ipynb`: The main entry point and benchmarking notebook.
*   `ddp_train.py`: The script launched by `torchrun` for the DDP experiment.
*   `train_utils.py`: Helper functions for model loading and training loops.

---

## Results (ResNet34)

| Mode   | Avg Epoch Time | Speedup vs Single | Scaling Efficiency |
| ------ | -------------- | ----------------- | ------------------ |
| Single | 6.302 s        | -                 | -                  |
| DP     | 12.460 s       | 0.51× (slower)    | 25% (inefficient)  |
| DDP    | 3.282 s        | 1.92× (faster)    | 96% (efficient)    |

The results were the same using [`ResNet18`](http://localhost:8888/notebooks/multiGPU_training.ipynb). 

---

## Summary of Findings

1. **DataParallel (DP) is inefficient for this workload:**
   DP resulted in about a **2× slowdown** compared to a single GPU.
   *Why?* DP uses a single process that scatters inputs, replicates the model across GPUs, and gathers outputs/gradients on a “master” GPU. For relatively lightweight models and inputs (like ResNet on CIFAR-10), this communication and coordination overhead outweighs the compute benefits.

2. **DistributedDataParallel (DDP) scales almost linearly:**
   DDP achieved a **1.92× speedup** on 2 GPUs (96% efficiency).
   *Why?* DDP spawns one process per GPU, avoiding Python GIL contention, and uses NCCL’s Ring-AllReduce to synchronize gradients efficiently. Models stay local to each GPU, and only gradients are communicated, which greatly reduces overhead compared to DP.

**Conclusion:**  
For multi-GPU training in PyTorch, **DistributedDataParallel is the recommended standard**. DataParallel generally adds unnecessary overhead and should be avoided for training workloads.
