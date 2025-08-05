---
title: "ML System Load Testing and Capacity Planning"
date: 2025-08-05
tags: ["mlops", "load-testing", "capacity-planning", "sre", "infrastructure", "kubernetes", "performance"]
categories: ["ML Engineering", "Infrastructure"]

---

## Introduction

In machine learning, creating a model with high accuracy is often celebrated as the finish line. But in reality, it's just the starting pistol for the race to production. A model that performs amazingly well in a Jupyter notebook can fail spectacularly under the chaotic, high-stakes conditions of live traffic. Imagine building a recommendation engine and it starts crashing during a holiday sale or your fraud detection API timing out on every transaction. This isn't just a technical glitch; it's a direct impact on business revenue and user trust.

This is where load testing and capacity planning become critical. Unlike traditional web services, ML systems have a unique and often demanding resource footprint. Their performance doesn't always scale linearly, and they rely heavily on specialized hardware like GPUs. A few years ago, deploying a model might have involved manually provisioning a large server and just "hoping for the best." Today, that's like navigating a ship in a storm without a compass.

This shift is driven by two things: real-time demand and economic pressure. Users expect instantaneous results, whether it's a language translation or an image generation. At the same time, the cloud resources needed to power these models, especially GPUs, are expensive. The goal is to provision just enough capacity to deliver a flawless user experience without burning a hole in your budget.

This has made systematic performance engineering an absolute necessity. The evolution was predictable:

1. Manual "Guesswork" Deployment: Provisioning a server based on intuition.
2. Basic Load Testing: Running a simple script to see "if it breaks."
3. Continuous Load Testing & Automated Capacity Planning: Integrating performance validation into the CI/CD pipeline and using data to drive automated scaling decisions.

The impact is profound. A well-planned system can serve millions of users reliably, while a poorly planned one becomes a constant source of outages and excessive costs. Proactive planning is the difference between a product that scales and a product that fails.

## Why is Load Testing ML Systems Different?

The goal of load testing is to understand how a system behaves under a specific load. But for ML inference services, the "load" and the "system" have unique characteristics that make them different from a standard REST API serving data from a database.

* **Payload Complexity:** The input to a model isn't always a small JSON object. It can be high-resolution images, long audio clips, or large blocks of text, which puts a strain on network I/O and memory before the model even starts its work.
* **Non-Linear Resource Profile:** A traditional API might see its CPU usage grow linearly with requests. An ML model's GPU or CPU usage might behave erratically depending on the input and techniques like batching.
* **The Latency vs. Throughput Trade-off:** This is a central challenge in ML serving. We can often increase throughput (predictions per second) by batching requests together, but this almost always comes at the cost of higher latency for each individual request. Finding the right balance is key.
* **Cold Starts:** When a new model instance starts up, it needs to load the model weights (which can be gigabytes) into memory (CPU and GPU). The first request to this "cold" instance can be orders of magnitude slower than subsequent requests.

| **Traditional Web Service** | **ML Inference Service** |
|:---------------------------|:-------------------------|
| Small JSON Request | Large Payload (Images/Text) |
| CPU/RAM Bound Processing | GPU/RAM Bound Processing |
| Fast Database Query | Model Loading (Cold Start) |
| Simple Response Logic | Inference + Batching |
| Small JSON Response | Processed Response |
| **Predictable resource usage** | **Variable resource patterns** |
| **Linear scaling** | **Non-linear scaling** |

## The Core Metrics: What Should We Measure?

To effectively test and plan, you need to measure what matters. Vanity metrics are useless; we need actionable data that maps directly to user experience and cost.

1. **Latency:** This is the time it takes for a single request to be processed. We almost always measure it in percentiles.

   * **p50 (Median):** Half of your users experience this latency or less.
   * **p90/p95:** Your power users' experience.
   * **p99/p99.9:** The "worst-case" experience. A high p99 latency means some users are having a very slow and frustrating time, even if the median looks good. This is often the most important metric for SLOs.

2. **Throughput:** The number of requests the system can handle per unit of time, usually measured in Queries Per Second (QPS) or Requests Per Second (RPS). This is a measure of a system's total capacity.

3. **Utilization:** How much of the provisioned resources are being used? This includes CPU, RAM, and most importantly for many models, GPU compute utilization and GPU memory utilization. Low utilization means you're wasting money; consistently high utilization (for example, >90%) means you have no headroom and are at risk of latency spikes and failures.

4. **Cost:** The ultimate business metric. A common way to normalize this is cost per million inferences. This allows you to compare the efficiency of different hardware, models, or configurations. A simplified formula is:

   Cost per 1M Inferences = (Instance Cost per Hour) / (QPS × 3600) × 1,000,000

{{<mermaid>}}
graph TD
 subgraph "Key Performance Indicators"
 A[Latency p99 less than 200ms]
 B[Throughput greater than 500 QPS]
 C[GPU Utilization 60 to 80 percent]
 D[Cost per 1M Inferences less than 50 cents]
 end
 classDef default fill:#2d2d2d,stroke:#ccc,stroke-width:2px,color:#fff
{{</mermaid>}}

## A Systematic Approach to Load Testing

A successful load testing strategy is a scientific process, not a random exploration. It involves a clear plan and methodical execution.

### The Process

1. **Define Service Level Objectives (SLOs):** What does "good performance" mean for your application? This is a contract with your users (and your business). Example: "The p99 latency for the text summarization API must be below 500ms while serving 100 QPS."
2. **Characterize the Workload:** Simulate realistic traffic. Are requests arriving in bursts or a steady stream? What is the distribution of payload sizes? Using production trace data to model your load is the gold standard.
3. **Choose Your Tools:** Use a modern load testing tool. K6 (JavaScript-based) and Locust (Python-based) are popular open-source choices that are developer-friendly and highly extensible.
4. **Execute and Monitor:** Run the tests against a production-like staging environment. As the load generator runs, watch your monitoring dashboards (for example, in Grafana) to see how the system's core metrics respond in real-time.

### Test Types

* **Capacity Test:** Start with low traffic and gradually ramp it up. Your goal is to find the maximum throughput the system can handle before one of your SLOs (usually p99 latency) is breached. This tells you the effective capacity of a single instance.
* **Soak Test:** Run a moderate, sustained load for a long period (hours or even days). This is excellent for detecting memory leaks or other performance degradation issues that only appear over time.
* **Stress Test:** Push the system far beyond its breaking point. What happens? Does it fail gracefully with HTTP 503 errors, or does it crash and burn? This tests the system's resilience.

{{<mermaid>}}
graph TD
 A[Load Tester] --> B[Start Capacity Test Plan]
 B --> C[Ramp Up Traffic]
 C --> D[Send Batch Requests]
 D --> E[ML Service Response]
 E --> F[Monitor Latency and Utilization]
 F --> G{Latency p99 exceeds SLO?}
 G -->|Yes| H[Stop Test]
 G -->|No| C
 H --> I[Analyze Results Find Max QPS]
 classDef default fill:#2d2d2d,stroke:#ccc,stroke-width:2px,color:#fff
{{</mermaid>}}

## From Testing to Capacity Planning

Load testing gives you the raw data. Capacity planning is the process of using that data to make informed architectural and financial decisions.

### Finding the "Knee" and Scaling

From your capacity test, you can plot latency against throughput. You will inevitably find a "knee" in the curve - the point where latency begins to shoot up exponentially. The QPS at this point is the maximum effective capacity of a single replica.

{{<mermaid>}}
graph TD
 A[Throughput vs Latency Analysis] --> B[Identify Knee Point]
 B --> C[Max Capacity approximately 65 QPS]
 C --> D[Calculate Required Replicas]
 D --> E[Target 500 QPS requires 8 replicas]
 E --> F[Add Buffer for Redundancy]
 classDef default fill:#2d2d2d,stroke:#ccc,stroke-width:2px,color:#fff
{{</mermaid>}}

With this number, you can plan your scaling strategy:

* **Horizontal Scaling (More Replicas):** This is the most common strategy. If your target is 500 QPS and you know one replica handles 65 QPS, you'll need at least ceil(500 / 65) = 8 replicas, plus some buffer for redundancy and traffic spikes. This is the foundation of autoscaling, where systems like Kubernetes' Horizontal Pod Autoscaler (HPA) automatically add or remove replicas based on observed metrics like CPU utilization.
* **Vertical Scaling (Bigger Machines):** Sometimes, adding more replicas doesn't help. If your model is too large to fit into the memory of a smaller GPU, you have no choice but to scale up to a larger instance (for example, from an NVIDIA T4 to an A100). Load testing different instance types is crucial for finding the most cost-effective option.

### Autoscaling on the Right Metrics

CPU-based autoscaling often works poorly for GPU-bound models. A model could be using 100% of its GPU while the CPU sits nearly idle. This is where custom metrics come in. Using tools like KEDA, you can autoscale based on more relevant signals like GPU utilization or even the number of messages in a request queue.

## How do we perform all these methods?

Thankfully, a mature ecosystem of tools exists to help you implement a robust testing and serving strategy.

| Tool/Framework | What It's For |
|:--------------|:-------------|
| **Locust / K6** | Modern, developer-friendly load generation tools. Perfect for defining your load tests as code and integrating them into CI/CD pipelines. |
| **Kubernetes / KServe** | The standard for orchestrating containerized applications. KServe (formerly KFServing) provides a production-ready serving layer on top of Kubernetes with features like autoscaling, traffic splitting, and canary deployments. |
| **Triton / TorchServe** | High-performance inference servers from NVIDIA and PyTorch, respectively. They offer critical optimizations out-of-the-box, such as dynamic batching and concurrent model execution, which are essential for maximizing throughput. |
| **Prometheus & Grafana** | The de-facto open-source stack for monitoring and observability. Scrape metrics from your servers and build dashboards to visualize your key performance indicators during load tests. |
| **NVIDIA Nsight Systems** | An advanced profiler for drilling deep into performance on the GPU. When you need to understand exactly where every millisecond is going inside the model's execution, this is the tool. |

## Conclusion

ML system load testing and capacity planning are no longer optional "nice-to-haves." They are core competencies of any team serious about running machine learning in production. Moving past the notebook means embracing the principles of SRE and performance engineering. By systematically measuring your system's limits, you can make data-driven decisions that balance cost, performance, and reliability.

The process is an iterative loop: deploy, test, analyze, plan, and repeat. It's the crucial bridge that turns a powerful model into a dependable and cost-effective product that delights users at scale.

## Further Reading

For those who want to dive deeper into the technical details and best practices, here are some excellent resources:

1. **Locust Documentation:** The official docs are a great place to start learning how to write your first load test. [Read the Docs](https://docs.locust.io/)
2. **K6 Documentation:** The official documentation for K6, another excellent load testing tool. [k6.io/docs](https://k6.io/docs/)
3. **NVIDIA Triton Inference Server Best Practices:** A guide from NVIDIA on how to get the maximum performance out of Triton, including details on dynamic batching and model configuration. [Read the Guide](https://www.google.com/search?q=https://github.com/triton-inference-server/server/blob/main/docs/user_guide/optimization.md)
4. **Google SRE Book - Chapter on Capacity Planning:** While not specific to ML, this chapter from the seminal book on Site Reliability Engineering covers the fundamental principles of capacity planning. [Read the Chapter](https://www.google.com/search?q=https://sre.google/sre-book/capacity-planning/)
5. **KServe Official Website:** Explore the documentation for KServe to understand how to build a modern, scalable inference platform on Kubernetes. [kserve.github.io](https://kserve.github.io/website/)
6. **"Operationalizing Machine Learning: An Interview Study"**: An insightful research paper that surveyed ML practitioners about the challenges they face in production, highlighting the importance of testing and monitoring. [Read the Paper on arXiv](https://arxiv.org/abs/2209.09125)