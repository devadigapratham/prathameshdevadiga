---
title: "Multi-GPU and Distributed Training Strategies"
date: 2025-07-19
tags: ["machine-learning", "distributed-systems", "gpu", "pytorch", "mlops", "deep-learning"]
categories: ["ML Engineering", "Infrastructure"]

---

## Introduction: The Unavoidable Rise of Distributed Training

In the world of machine learning, it feels like we're living in a shell due to all the new advancements that _literally_ happen every week. Models like Gemini 2.5 Pro, Kimi K2, GPT-4o and Stable Diffusion aren't just small improvements; they're a massive leap in capability which in turn is also an increase in scale. A few years ago, training a top model on a single, powerful GPU was normal. Today, that’s like trying to build a skyscraper with a single crane. It’s simply not possible.

This shift is driven by two things: **model size** and **dataset size**. Modern AI models have exploded from millions to hundreds of billions to even **trillions** of parameters. A model like GPT-3, for example, needs about 350 GB of memory just for its parameters which is far more than any single GPU can handle. At the same time, the datasets we use to train them have grown to terabytes of text and images.

This has made **distributed training**, the art of splitting the work across many processors as an absolute **necessity**. The evolution was quick:
1.  **Single GPU:** The starting point.
2.  **Multi-GPU (Single Machine):** The first step into parallel work.
3.  **Distributed Training (Multiple Machines):** Using a whole cluster of computers. This is the supercomputing power once reserved for governments, now being used for AI.

The impact is huge. Training a model on the ImageNet dataset once took weeks; now, it can be done in minutes. _Faster training means faster research, faster products, and a stronger competitive edge._

***

## The Two Major Types of Parallelism

The goal is to break down the massive job of training a neural network into smaller pieces that can be worked on at the same time. The main challenge is deciding *how* to split the job and how the different workers (GPUs) communicate. There are two fundamental strategies: **Data Parallelism** and **Model Parallelism**.

### Data Parallelism: More Workers, Same Blueprint

This is the most common and intuitive strategy. The idea is simple: you have one model blueprint, but you give each worker a different slice of the data to work on.

It works like this:
1.  **Replicate:** A complete copy of the model is loaded onto every GPU.
2.  **Shard:** The big batch of training data is split into smaller *micro-batches* and each GPU gets one.
3.  **Process:** Each GPU independently processes its data, calculating the necessary updates (gradients).
4.  **Sync Up:** This is the key communication step. All GPUs share their results and average them together. This ensures every model copy stays identical. This is usually done with an operation called `AllReduce`.
5.  **Update:** Each GPU uses the averaged result to update its copy of the model.

Since every GPU ends up with the exact same model weights after each step, they stay perfectly in sync.

{{<mermaid>}}
graph TD
    subgraph "Worker 1 (GPU 0)"
        M0[Model Replica] -->|Data Slice 0| FP0[Forward & Backward Pass]
        FP0 --> G0[Local Gradients 0]
    end
    subgraph "Worker 2 (GPU 1)"
        M1[Model Replica] -->|Data Slice 1| FP1[Forward & Backward Pass]
        FP1 --> G1[Local Gradients 1]
    end
    subgraph "Worker N (GPU N)"
        MN[Model Replica] -->|Data Slice N| FPN[Forward & Backward Pass]
        FPN --> GN[Local Gradients N]
    end

    G0 --> AR((AllReduce Sync))
    G1 --> AR((AllReduce Sync))
    GN --> AR((AllReduce Sync))

    AR -->|Averaged Gradients| U0[Update Model 0]
    AR -->|Averaged Gradients| U1[Update Model 1]
    AR -->|Averaged Gradients| UN[Update Model N]
    
    classDef default fill:#2d2d2d,stroke:#ccc,stroke-width:2px,color:#fff
    classDef syncNode fill:#422,stroke:#e55,stroke-width:2px,color:#fff
    classDef updateNode fill:#242,stroke:#5e5,stroke-width:2px,color:#fff

    class AR syncNode
    class U0,U1,UN updateNode
{{</mermaid>}}

The main limitation? **The entire model must fit on a single GPU.** When your model gets too big, this strategy alone isn't enough.

### Model Parallelism: One Blueprint, Many Specialists

When a model is too large for one GPU, you have to split the model itself. Instead of every worker doing the same job on different data, each worker becomes a specialist responsible for just one part of the model.

#### Pipeline Parallelism: The Assembly Line

In this approach, you assign consecutive layers of the model to different GPUs. Think of it like this:
* GPU 0 handles the first few layers.
* GPU 1 handles the next few layers.
* GPU 2 handles the final layers.

The output of one GPU becomes the input for the next.

{{<mermaid>}}
%%{init: {'theme': 'dark'}}%%
graph TD
    Input --> GPU0[Stage 0: Layers 1-8]
    GPU0 -->|Activations| GPU1[Stage 1: Layers 9-16]
    GPU1 -->|Activations| GPU2[Stage 2: Layers 17-24]
    GPU2 -->|Activations| GPU3[Stage 3: Layers 25-32]
    GPU3 --> Output
{{</mermaid>}}

A simple implementation of this can be inefficient, as GPUs have to wait for the one before them to finish. We'll see how to fix this "bubble" of idle time later.

#### Tensor Parallelism: Teamwork Within a Layer

This is a more fine-grained approach where you split up the work *inside* a single large layer. Imagine a huge calculation within one layer. Instead of one GPU doing all the math, you can split the calculation across several GPUs. They each do a piece, then combine their results. This requires extremely fast connections between the GPUs (like NVIDIA's NVLink) because they need to communicate constantly.

### Hybrid Approaches: The Best of All Worlds

To train *extremely huge* models, you need to combine everything. This is often called **3D Parallelism**:

1.  **Pipeline Parallelism (PP):** Splits the model into an assembly line of stages.
2.  **Tensor Parallelism (TP):** Splits the work *within* each stage of the assembly line.
3.  **Data Parallelism (DP):** Runs multiple copies of this entire assembly line, each on different data.

This hybrid strategy is the key that unlocks training for models with hundreds of billions or even trillions of parameters.

{{<mermaid>}}
%%{init: {'theme': 'dark'}}%%
graph TD
    subgraph DP0 [Data Parallel Group 0]
        direction LR
        subgraph PS0_0 [Pipeline Stage 0]
            TP0_0[GPU 0] --"NVLink (TP)"--> TP0_1[GPU 1]
        end
        subgraph PS1_0 [Pipeline Stage 1]
            TP1_0[GPU 2] --"NVLink (TP)"--> TP1_1[GPU 3]
        end
        PS0_0 --"Network (PP)"--> PS1_0
    end

    subgraph DP1 [Data Parallel Group 1]
        direction LR
        subgraph PS0_1 [Pipeline Stage 0]
            TP1_0_0[GPU 4] --"NVLink (TP)"--> TP1_0_1[GPU 5]
        end
        subgraph PS1_1 [Pipeline Stage 1]
            TP1_1_0[GPU 6] --"NVLink (TP)"--> TP1_1_1[GPU 7]
        end
        PS0_1 --"Network (PP)"--> PS1_1
    end

    DP0 <--"AllReduce (DP)"--> DP1
{{</mermaid>}}

***

## How does Communication occur?

In distributed training, computation is only half the story. The other half is communication. Moving huge amounts of data between GPUs efficiently is critical.

### The AllReduce Algorithm

The foundation of data parallelism is `AllReduce`. Its goal is to take numbers from all workers, combine them (for example, average them), and give the final result back to everyone. The most popular way to do this is with **Ring AllReduce**.

Imagine the GPUs are sitting in a circle. Each GPU passes a piece of its data to its neighbor. This happens over and over, with each GPU both sending and receiving data at the same time. It's like a highly efficient game of "whisper down the lane" for math. After a series of steps, every GPU ends up with the final, averaged result. This method is brilliant because it uses the network bandwidth incredibly well.

{{<mermaid>}}
%%{init: {'theme': 'dark'}}%%
graph TD
    subgraph Ring Communication
        GPU0 <-->|Chunk| GPU1
        GPU1 <-->|Chunk| GPU2
        GPU2 <-->|Chunk| GPU3
        GPU3 <-->|Chunk| GPU0
    end
{{</mermaid>}}

### Overlapping Computation and Communication

A key trick is to hide the time it takes to communicate by doing it at the same time as computation. During the training step where the model calculates its updates (the "backward pass"), the updates for the last layers are ready first. You don't have to wait for the entire process to finish.

Modern tools can kick off the `AllReduce` for these ready updates immediately, while the GPU works on calculating updates for the earlier layers. This "pipelining" of work is like a chef starting to chop the next vegetable while the first one is already on the stove. It's essential for keeping the expensive GPUs busy and not waiting around.

{{<mermaid>}}
%%{init: {'theme': 'dark'}}%%
sequenceDiagram
    participant BWD as Backward Pass
    participant COMM as Communication
    BWD->>BWD: Compute Grads for Layer N
    BWD->>+COMM: Start AllReduce for Layer N Grads
    BWD->>BWD: Compute Grads for Layer N-1
    Note right of COMM: Overlap!
    BWD->>+COMM: Start AllReduce for Layer N-1 Grads
    deactivate COMM
    deactivate COMM
{{</mermaid>}}

***

## Advanced Techniques for Massive Scale

Let's now understand more methods that make training giant models possible.

### ZeRO: The Zero Redundancy Optimizer

Developed by Microsoft, ZeRO is a "memory diet" for data parallelism. In standard data parallelism, every GPU wastefully holds a full copy of the model parameters, the gradients (updates), and the optimizer states (extra data used by modern optimizers). For large models, this is a huge amount of redundant memory.

ZeRO cleverly partitions these things across the GPUs, so each worker only holds a slice of the total.
* **Stage 1:** Partitions the optimizer states.
* **Stage 2:** Also partitions the gradients.
* **Stage 3:** Partitions the model parameters themselves.

With ZeRO Stage 3, each GPU only needs to hold the part of the model it's currently working on. It fetches the next piece it needs just in time from its peers. This dramatically reduces the memory needed per GPU, making it possible to train models with trillions of parameters on existing hardware.

{{<mermaid>}}
graph TD
    subgraph "Standard DDP Memory (per GPU)"
        direction LR
        P1[Parameters]
        G1[Gradients]
        O1[Optimizer States]
    end
    subgraph "ZeRO-1 (per GPU)"
        direction LR
        P2[Parameters]
        G2[Gradients]
        O2_part[Partitioned<br>Optimizer States]
    end
    subgraph "ZeRO-2 (per GPU)"
        direction LR
        P3[Parameters]
        G3_part[Partitioned<br>Gradients]
        O3_part[Partitioned<br>Optimizer States]
    end
    subgraph "ZeRO-3 (per GPU)"
        direction LR
        P4_part[Partitioned<br>Parameters]
        G4_part[Partitioned<br>Gradients]
        O4_part[Partitioned<br>Optimizer States]
    end
    
    classDef default fill:#2d2d2d,stroke:#ccc,stroke-width:2px,color:#fff
{{</mermaid>}}

### Mixed Precision and Gradient Accumulation

These two tricks are used almost universally.
* **Gradient Accumulation:** Lets you simulate a much larger training batch than can fit in memory. You simply process several small batches, adding up their updates, and only apply the combined update at the end. It's like pretending you have a bigger backpack by making multiple trips.
* **Automatic Mixed Precision (AMP):** Training with full 32-bit numbers is memory-hungry and slow. AMP uses a mix of 16-bit and 32-bit numbers. Using 16-bit numbers is like using shorthand instead of full sentences, hence it makes it much faster and uses half the memory, and for neural networks, it's usually "good enough." This can provide a huge speedup on modern GPUs with specialized hardware (Tensor Cores).

### Fixing the Pipeline "Bubble"

Remember the assembly line analogy for pipeline parallelism, where GPUs might sit idle? The solution is to split the data batch into many tiny **micro-batches**. The pipeline stages then process these micro-batches in a staggered fashion. As soon as the first micro-batch is done with stage 1, it moves to stage 2, and stage 1 can immediately start on the second micro-batch. This keeps all the GPUs working most of the time, dramatically improving efficiency.

{{<mermaid>}}
%%{init: {'theme': 'dark'}}%%
gantt
    title Pipeline Execution Schedule (Showing Bubbles)
    dateFormat S
    axisFormat %Ss
    
    section GPU 0 (Stage 0)
    Forward MB 1 : 0, 2
    Forward MB 2 : 2, 2
    Forward MB 3 : 4, 2
    Backward MB 3 : 6, 2
    Backward MB 2 : 8, 2
    Backward MB 1 : 10, 2
    
    section GPU 1 (Stage 1)
    Bubble (Idle) : 0, 2
    Forward MB 1 : 2, 2
    Forward MB 2 : 4, 2
    Forward MB 3 : 6, 2
    Bubble (Idle) : 8, 2
    Backward MB 2 : 10, 2
    
    section GPU 2 (Stage 2)
    Bubble (Idle) : 0, 4
    Forward MB 1 : 4, 2
    Forward MB 2 : 6, 2
    Bubble (Idle) : 8, 4
    Backward MB 1 : 12, 2
{{</mermaid>}}
***

## How do we perform all these methods tho?

Thankfully, powerful libraries handle most of the complexity for you.

| Framework       | What It's For                                                                                                                                                             |
| :-------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **PyTorch DDP** | The standard, built-in way to do data parallelism in PyTorch. It's easy to use and very efficient for most cases.                                                           |
| **Horovod** | An open-source library from Uber that makes data parallelism easy across different frameworks (like TensorFlow and PyTorch).                                                  |
| **DeepSpeed** | A library from Microsoft that provides an all-in-one system for training massive models. It's the easiest way to use advanced techniques like ZeRO and pipeline parallelism. You just write a simple configuration file to enable them. |

The key to good performance is finding the bottleneck. Is your training slow because of data loading, network communication, or the computation itself? Tools like the **PyTorch Profiler** can give you a detailed timeline of what's happening, showing you exactly where the slowdowns are. If you see GPUs sitting idle, it means you have a bottleneck somewhere that needs fixing.

{{<mermaid>}}
%%{init: {'theme': 'dark'}}%%
gantt
    title Profiler View: Bottleneck Analysis
    dateFormat  X
    axisFormat %s

    section Good Overlap
    GPU Compute      : 0, 4
    GPU Communication  : 2, 4
    Optimizer Step : 4, 5

    section Communication Bottleneck
    GPU Compute      : 6, 10
    GPU Communication  : 8, 12
    Idle (Waiting) : crit, 10, 2
    Optimizer Step : 12, 1
{{</mermaid>}}

***

## Conclusion

Distributed training is a fascinating field that blends machine learning with high-performance computing. While it seems complex, the core ideas of data, model, and pipeline parallelism are the foundation for nearly all large-scale training. Frameworks like PyTorch and DeepSpeed have made these powerful techniques accessible to everyone, not just giant tech companies.

***

## Further Reading  

For those who want to dive deeper into the technical details and the foundational research, here are some excellent resources:

1.  **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models:** The original paper from Microsoft that introduced the ZeRO optimizer, a cornerstone of modern large-scale training. A must-read for understanding memory optimization. [Read the Paper on arXiv](https://arxiv.org/abs/1910.02054)
2.  **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism:** The NVIDIA paper that popularized tensor and pipeline parallelism for large language models. [Read the Paper on arXiv](https://arxiv.org/abs/1909.08053)
3.  **DeepSpeed Official Website:** The best place to find documentation, tutorials, and blog posts about the DeepSpeed library, which implements ZeRO and other advanced techniques. [deepspeed.ai](https://www.deepspeed.ai/)
4.  **PyTorch DistributedDataParallel (DDP) Documentation:** The official documentation is the source of truth for implementing data parallelism in PyTorch. [Read the Docs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
5.  **Horovod: fast and easy distributed deep learning:** The paper introducing Uber's popular framework, which made distributed training much more accessible. [Read the Paper on arXiv](https://arxiv.org/abs/1802.05799)
6.  **GPipe: Efficient Training of Giant Neural Networks:** Google's paper on pipeline parallelism, which laid the groundwork for many modern pipeline scheduling techniques. [Read the Paper on arXiv](https://arxiv.org/abs/1811.06965)