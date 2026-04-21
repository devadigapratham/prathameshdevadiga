+++
title = "About"
slug = "about"
+++

I'm Prathamesh Devadiga — incoming CS PhD student at Dartmouth College, where I'll be working with [Prof. Shawn Shan](https://www.cs.dartmouth.edu/~sshan/) on machine learning security and LLM privacy. I'm finishing my B.Tech in Computer Science at PES University, Bangalore (GPA: 8.91/10).

My research sits at the intersection of ML security, production AI systems, and trustworthy AI. The core question I keep coming back to: *what does a model actually know, and what are the limits of our ability to control that?*

I also lead **Ādhāra AI Labs**, an independent research lab working on everything from compiler optimization with small language models to efficient neural architectures for resource-constrained environments. Our work has appeared at NeurIPS, ICIAI, ICML, and various workshops.


## What I'm Working On

**LLM Memorization & Data Extraction** — With Prof. Shan at Dartmouth, I built Hierarchical Extraction Search (HES), a query-efficient framework that extracts memorized training data from LLMs at 10–100× lower cost than brute-force sampling, demonstrated across 12 state-of-the-art models (7B to 123B parameters). The paper is under review at ICML 2026. The empirical result is clean; the theoretical question underneath it — why certain prompt structures unlock memorized content while nearly identical ones don't — is what I'm pursuing in the PhD.

**World Models in Transformers** — At Lossfunk, I'm investigating how to induce robust world models in Transformers by modifying training objectives and architectural biases. The goal is to mitigate structural decay in learned representations and force models toward stable, causally grounded internal states through long-horizon future prediction and action-conditioned counterfactuals.

**Low-Resource Language Modeling** — I've built systems for Tulu (~0.001% of typical training data), where hard negative constraints reduced catastrophic language leakage from 80% to 5%. The insight: explicit prohibitions outperform positive instructions for maintaining language integrity. Under review at EACL 2025 LoResLM Workshop.

**Production AI Systems** — I build end-to-end LLM training and serving infrastructure: distributed training frameworks (FSDP/DDP), real-time jailbreak prevention under 5ms latency, and scalable serving with quantization and speculative decoding.


## Background

B.Tech, Computer Science — PES University, Bangalore (GPA: 8.91/10, 2022–2026). Teaching Assistant for ML and Deep Learning, mentoring 75+ students. Previously: Research Assistant at Dartmouth College, AI Research Intern at Lossfunk, Nokia (Intent-Based Network Management), Google Summer of Code 2025 at UC Santa Cruz, and Undergraduate Research Intern at IIT Indore.

Recognition: Amazon AI-ML Scholar (top 1,000 in India), Cisco ThingQbator Hackathon winner, Oxford Machine Learning School, Cohere AI Summer School, 6× Merit Scholarships at PES University.


## Beyond Research

I've co-instructed a 30-hour Deep Learning course, served as Head of Technology for the PES Entrepreneurship Club, mentored 50+ teams in hackathons and the WiDS Datathon, and delivered talks on DSPy and LLM fine-tuning at FOSS United and GDSC events. I believe in open-source research and making AI accessible beyond well-resourced labs.


## Research Interests

- **ML Security & LLM Privacy:** Data extraction, memorization leakage, adversarial robustness, jailbreak defenses
- **Alignment Safety:** Privacy-alignment tradeoffs, safety training vulnerabilities, constraint-based defenses
- **World Models & Representation Learning:** Causal grounding in Transformers, structural representation stability
- **Production AI Systems:** Low-latency inference, distributed training, efficient serving infrastructure
- **Low-Resource NLP:** Extremely low-resource languages, structured learning, Indic LLMs


*Starting PhD at Dartmouth in fall 2026. Reach out via [email](mailto:devadigapratham8@gmail.com) or [LinkedIn](https://linkedin.com/in/prathamesh-devadiga).*
