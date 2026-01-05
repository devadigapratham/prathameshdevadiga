+++
title = "Projects"
slug = "projects"
+++

Here are key projects that reflect my work across machine learning systems, generative AI, security, and applied research. Each project represents a unique challenge solved through innovative engineering and research.

---

## **AI & Machine Learning Projects**

### **PSRG PR Agent** | *Multi-Agent GitHub PR Reviewer*

A real-time, multi-agent system using LangGraph to automate GitHub Pull Request reviews. The system features a novel PR-Aware Self-Referee Graph (PSRG) architecture that enables high-fidelity code analysis and intelligent patch suggestions. All agents are powered exclusively by open-source, ≤1B parameter models, demonstrating efficient AI without relying on large proprietary models.

**Key Features:**
- Real-time PR analysis and review generation
- Multi-agent coordination for comprehensive code review
- Intelligent patch suggestion based on code context
- Lightweight deployment using small language models

**Tech Stack:** LangGraph, Multi-Agent Systems, LLMs, GitHub API

---

### **JurisQwen** | *Domain-Specific Legal AI*

A production-ready legal AI system fine-tuned on Indian law datasets. The system leverages Qwen2.5-7B with LoRA fine-tuning, optimized using PEFT and Unsloth for efficient training. Deployed with 4-bit quantization and Flash Attention 2 on Modal platform, achieving significant cost reduction while maintaining high accuracy for legal document analysis and question-answering.

**Key Features:**
- Specialized fine-tuning on Indian legal corpus
- Efficient inference with 4-bit quantization
- Scalable deployment on cloud infrastructure
- Fast response times with Flash Attention optimization

**Tech Stack:** LoRA, Modal, Quantization, Qwen2.5-7B, Indian Legal Dataset

---

### **CoDSPy** | *AI-Powered Code Optimization System*

An intelligent code analysis platform that combines Chain-of-Thought (CoT) and ReAct reasoning with local LLM technologies. The system provides autonomous code optimization, comprehensive syntax inspection, and automated test case generation, making it a powerful tool for developers seeking to improve code quality and performance.

**Key Features:**
- Autonomous code optimization suggestions
- Advanced syntax and pattern analysis
- Automated test case generation
- Local LLM integration for privacy and cost efficiency

**Tech Stack:** Python, DSPy, Gradio, Local LLMs, CoT Reasoning

---

### **KASPER** | *PDF Malware Detection System*

A state-of-the-art deep learning architecture for detecting PDF-based malware with exceptional adversarial robustness. The system achieves 99.5% accuracy through innovative custom malware injection pipelines and spline-based kernel layers, demonstrating strong resilience against sophisticated adversarial attacks.

**Key Features:**
- 99.5% detection accuracy on PDF malware
- Adversarial robustness through custom defense mechanisms
- Custom malware injection pipeline for training
- Spline-based kernel layers for enhanced feature extraction

**Tech Stack:** Python, PyTorch, Adversarial ML, Custom Malware Pipeline

---

### **Cerebrum** | *Production LLM Training and Serving System*

An end-to-end large language model system featuring distributed FSDP/DDP training, Flash Attention v2, and Mixture-of-Experts (MoE) support. The system includes Mixture-of-Refusals (MoR), a novel defense mechanism achieving 2-3x speedup on safe queries while maintaining identical safety guarantees through conditional safety routing. Features a high-performance vLLM-based inference engine with quantization, speculative decoding, and prefix caching, deployed with complete infrastructure including Docker, Kubernetes, and Prometheus for scalable monitoring and fault-tolerant operations.

**Key Features:**
- Distributed FSDP/DDP training for large-scale model training
- Mixture-of-Refusals (MoR) defense with 2-3x speedup on safe queries
- High-performance vLLM inference with quantization and speculative decoding
- Production-ready infrastructure with Kubernetes and Prometheus monitoring
- Flash Attention v2 and Mixture-of-Experts support

**Tech Stack:** Python, PyTorch, vLLM, FSDP, Kubernetes, Prometheus

---

## **Distributed Systems & Infrastructure**

### **Arcane ML** | *Distributed ML Framework*

A production-ready framework for distributed machine learning training that seamlessly works across SSH clusters, Modal Cloud GPUs, and local multi-GPU setups. The framework provides comprehensive support for PyTorch DDP, enabling efficient multi-GPU scaling with automatic gradient synchronization. Features a unified CLI that abstracts away the complexity of distributed training workflows, making it accessible to researchers and engineers.

**Key Features:**
- Multi-environment support (SSH clusters, cloud, local)
- PyTorch DDP integration with automatic configuration
- Unified CLI for simplified distributed training
- Efficient gradient synchronization and communication

**Tech Stack:** Python, PyTorch DDP, Distributed Systems, Modal Cloud

---

### **goDFS** | *Decentralized File Storage System*

A fully distributed, content-addressable file storage system built in Golang. Designed from the ground up to handle and stream very large files efficiently across distributed nodes, providing fault tolerance and high availability through decentralized architecture.

**Key Features:**
- Content-addressable storage for deduplication
- Efficient streaming of large files
- Decentralized architecture for fault tolerance
- High-performance concurrent operations

**Tech Stack:** Go, Distributed Systems, Content-Addressable Storage

---

### **makedis** | *Modern Deployment Framework*

A modern deployment framework designed to streamline application deployments with industry-standard practices. The framework enhances static asset delivery through CDN integration and enables scalable reverse proxy configurations, supporting robust production deployments.

**Key Features:**
- Automated deployment pipelines
- Enhanced static asset delivery
- Scalable reverse proxy configurations
- Industry-standard deployment practices

**Tech Stack:** Go, Reverse Proxy, Static Asset Delivery, Deployment Automation

---

## **Data Processing & Analytics**

### **E-Commerce Real-Time Analytics** | *Big Data Processing Pipeline*

A high-performance real-time sales analytics system capable of processing high-velocity financial data streams. Built on Apache Flink for stream processing, with PostgreSQL for persistent storage and Elasticsearch for fast search and analytics. The entire system is orchestrated using Docker Compose, enabling scalable and maintainable deployments.

**Key Features:**
- Real-time stream processing with Apache Flink
- High-velocity data ingestion and processing
- Fast search capabilities with Elasticsearch
- Scalable containerized deployment

**Tech Stack:** Apache Flink, PostgreSQL, Elasticsearch, Docker Compose

---

### **RSServe** | *RSS Aggregator*

A fully-featured RSS aggregator built in Go that efficiently collects, parses, and organizes RSS feeds from multiple sources. The system provides a seamless reading experience with intelligent feed management and content organization.

**Key Features:**
- Efficient RSS feed collection and parsing
- Multi-source feed aggregation
- Intelligent content organization
- High-performance concurrent processing

**Tech Stack:** Go, RSS Processing, Web Scraping

---

## **Game AI & Algorithms**

### **Daisy** | *AlphaZero Implementation*

A complete implementation of the AlphaZero algorithm from scratch, demonstrating deep reinforcement learning and Monte Carlo Tree Search (MCTS) techniques. The system achieves superhuman performance in board games through self-play training and neural network-guided search.

**Key Features:**
- Complete AlphaZero algorithm implementation
- Self-play training mechanism
- Neural network-guided MCTS
- Superhuman game-playing performance

**Tech Stack:** Python, Deep Learning, Monte Carlo Tree Search, Game AI

---

## **Automation & Productivity Tools**

### **clutchCV** | *LinkedIn Job Search AI Agent*

An intelligent AI agent that enhances job searching on LinkedIn by automating the discovery and matching process. The system analyzes user resumes and experience to find suitable job matches, significantly reducing the time spent on manual job searching.

**Key Features:**
- Automated job discovery and matching
- Resume-based intelligent matching
- LinkedIn API integration
- Personalized job recommendations

**Tech Stack:** Python, LinkedIn API, AI Agent, Job Matching Algorithm

---

## **Open Source Contributions**

### **Billion-Scale Vector Embeddings Dataset** | *Google Summer of Code 2025, UC Santa Cruz*

A comprehensive billion-scale vector embedding dataset constructed from open-source codebases using open-source models. The dataset is designed for benchmarking Approximate Nearest Neighbor (ANN) algorithms, with embeddings of 768, 1024, and 2048 dimensions to reflect modern workloads. This work addresses critical limitations of existing benchmarks, enabling robust evaluations of vector search algorithms at scale.

**Key Features:**
- Billion-scale embedding dataset
- Multiple dimension variants (768, 1024, 2048)
- Realistic benchmarks from open-source codebases
- Comprehensive ANN algorithm evaluation framework

**Tech Stack:** Python, Open-source LLMs, Vector Databases, ANN Algorithms

---

*For more details about any project or collaboration opportunities, feel free to [reach out](/contact/).*  
