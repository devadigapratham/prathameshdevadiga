+++
title = "Projects"
slug = "projects"
+++

Selected projects across ML systems, distributed infrastructure, and applied AI research. Full list on [GitHub](https://github.com/devadigapratham).


<div class="project-section-label">ML Systems &amp; Research</div>

<div class="project-grid">

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">Cerebrum</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Production LLM Training &amp; Serving</div>
  <div class="project-card-desc">
    End-to-end LLM system with distributed FSDP/DDP training, Flash Attention v2, and Mixture-of-Experts support. Includes <strong>Mixture-of-Refusals (MoR)</strong>, a novel safety routing mechanism achieving 2–3× speedup on safe queries with identical safety guarantees. vLLM inference engine with quantization, speculative decoding, and prefix caching, deployed on Kubernetes with Prometheus monitoring.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">PyTorch</span>
    <span class="project-card-tag">FSDP/DDP</span>
    <span class="project-card-tag">vLLM</span>
    <span class="project-card-tag">Kubernetes</span>
    <span class="project-card-tag">Flash Attention</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">Arcane ML</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/arcane-ml" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Distributed Training Framework</div>
  <div class="project-card-desc">
    Production-ready framework for distributed ML training across SSH clusters, Modal Cloud GPUs, and local multi-GPU setups. PyTorch DDP with automatic gradient synchronization, unified CLI abstracting the complexity of distributed workflows.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">Python</span>
    <span class="project-card-tag">PyTorch DDP</span>
    <span class="project-card-tag">Modal Cloud</span>
    <span class="project-card-tag">Distributed Systems</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">KASPER</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">PDF Malware Detection · IIT Indore · Applied Soft Computing (Q1)</div>
  <div class="project-card-desc">
    Deep learning framework for PDF malware detection with 99.5% accuracy, robust against FGSM and PGD adversarial attacks. Custom malware injection pipeline for training; explainability via Kolmogorov-Arnold Networks. Published in <em>Applied Soft Computing</em> (Q1 Journal).
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">PyTorch</span>
    <span class="project-card-tag">Adversarial ML</span>
    <span class="project-card-tag">KANs</span>
    <span class="project-card-tag">Security</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">JurisQwen</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/JurisQwen" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Legal Domain LLM</div>
  <div class="project-card-desc">
    Qwen2.5-7B fine-tuned on Indian legal datasets using LoRA + PEFT + Unsloth. Deployed with 4-bit quantization and Flash Attention 2 on Modal. Specialized for Indian legal document analysis and question-answering.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">LoRA</span>
    <span class="project-card-tag">Qwen2.5-7B</span>
    <span class="project-card-tag">Quantization</span>
    <span class="project-card-tag">Modal</span>
    <span class="project-card-tag">Legal AI</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">CoDSPy</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/CoDSPy" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">AI-Powered Code Optimization</div>
  <div class="project-card-desc">
    Code optimization system using Chain-of-Thought and ReAct reasoning with local LLMs. Autonomous refactoring, syntax analysis, and automated test generation, fully local, no API costs.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">DSPy</span>
    <span class="project-card-tag">CoT Reasoning</span>
    <span class="project-card-tag">Gradio</span>
    <span class="project-card-tag">Local LLMs</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">Attention Rollout Live</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/attention-rollout" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Transformer Attention Visualizer · Apple Silicon</div>
  <div class="project-card-desc">
    Interactive visualizer that animates transformer attention weights in real time as a local LLM generates text. Every new token shows which prior tokens the model attended to, across all 28 layers and 12 heads. Live heatmap, per-layer scrubber with entropy sparkline, and per-head selection. No cloud API required.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">PyTorch</span>
    <span class="project-card-tag">FastAPI</span>
    <span class="project-card-tag">React</span>
    <span class="project-card-tag">D3</span>
    <span class="project-card-tag">SSE</span>
    <span class="project-card-tag">Apple Silicon</span>
  </div>
</div>

</div>

<div class="project-section-label">Distributed Infrastructure</div>

<div class="project-grid">

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">goDFS</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/goDFS" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Decentralized File Storage System</div>
  <div class="project-card-desc">
    Fully decentralized, content-addressable file storage system in Go. Handles streaming of large files across distributed nodes with fault tolerance through decentralized architecture and high-performance concurrent operations.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">Go</span>
    <span class="project-card-tag">Content-Addressable Storage</span>
    <span class="project-card-tag">Distributed Systems</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">Raft3D</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/raft3d" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Distributed 3D Printer Management</div>
  <div class="project-card-desc">
    Distributed 3D printer management system using the Raft Consensus Algorithm for data persistence, replacing traditional centralized databases with a consensus-based distributed log.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">Go</span>
    <span class="project-card-tag">Raft Consensus</span>
    <span class="project-card-tag">Distributed Systems</span>
  </div>
</div>

</div>

<div class="project-section-label">Open Source</div>

<div class="project-grid">

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">Billion-Scale Vector Embeddings Benchmark</div>
    <div class="project-card-links">
      <a href="https://github.com/ucsc-ospo" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">Google Summer of Code 2025 · UC Santa Cruz</div>
  <div class="project-card-desc">
    Billion-scale vector embedding benchmarks (768, 1024, 2048 dimensions) built from open-source codebases using open-source models. Addresses critical limitations of existing ANN benchmarks, enabling robust evaluation under realistic workloads. Selected for GSoC 2025 (acceptance rate &lt;10%).
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">Python</span>
    <span class="project-card-tag">Vector Search</span>
    <span class="project-card-tag">ANN Algorithms</span>
    <span class="project-card-tag">Benchmarking</span>
  </div>
</div>

<div class="project-card">
  <div class="project-card-header">
    <div class="project-card-title">Daisy</div>
    <div class="project-card-links">
      <a href="https://github.com/devadigapratham/Daisy" target="_blank" rel="noopener" class="project-card-link">GitHub ↗</a>
    </div>
  </div>
  <div class="project-card-label">AlphaZero from Scratch</div>
  <div class="project-card-desc">
    Complete AlphaZero implementation from scratch: self-play training, neural network-guided Monte Carlo Tree Search, achieving superhuman board game performance.
  </div>
  <div class="project-card-tags">
    <span class="project-card-tag">Python</span>
    <span class="project-card-tag">Deep RL</span>
    <span class="project-card-tag">MCTS</span>
    <span class="project-card-tag">Game AI</span>
  </div>
</div>

</div>
