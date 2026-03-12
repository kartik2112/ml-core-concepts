# ML Core Concepts - Interview Preparation Notebooks

A comprehensive repository of Jupyter notebooks and one-pagers covering essential machine learning and deep learning concepts for interview preparation.

## 📚 Contents

### 🎯 Theoretical Core Concepts — Interview Q&A (NEW!)
Comprehensive interview preparation notebooks with densely packed Q&A covering popular and tricky questions for AI/ML, Research Scientist, and ML Engineer roles. See [`theoretical-core-concepts/`](theoretical-core-concepts/) for:

| # | Notebook | Topics |
|---|---|---|
| 1 | [XGBoost Interview Q&A](theoretical-core-concepts/01_xgboost_interview_qa.ipynb) | Objective function, gain formula, L1/L2 reg, missing values, feature importance (weight/gain/cover/SHAP), LightGBM vs CatBoost, hyperparameter tuning, monotone constraints |
| 2 | [Random Forest & Ensemble Q&A](theoretical-core-concepts/02_random_forest_interview_qa.ipynb) | Bagging, OOB error, MDI vs permutation importance, bias-variance tradeoff, Extra Trees, Isolation Forest, boosting vs bagging vs stacking |
| 3 | [Deep Learning Concepts Q&A](theoretical-core-concepts/03_deep_learning_concepts_interview_qa.ipynb) | Backprop, vanishing gradients, activations (ReLU/GELU/SiLU), Adam/AdamW/SGD, weight decay vs L2, transformers vs RNNs, residual connections, dropout, scaling laws, double descent |
| 4 | [Normalization Types Q&A](theoretical-core-concepts/04_normalization_types_interview_qa.ipynb) | BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, SpectralNorm, AdaLN, Pre-LN vs Post-LN, train vs inference behavior |
| 5 | [RAG Types Q&A](theoretical-core-concepts/05_rag_types_interview_qa.ipynb) | Naive RAG, Advanced RAG, HyDE, hybrid search, reranking, GraphRAG, Self-RAG, CRAG, FLARE, Modular RAG, Agentic RAG, RAGAS evaluation |
| 6 | [Traditional ML Q&A](theoretical-core-concepts/06_traditional_ml_interview_qa.ipynb) | Bias-variance, SVM (kernel trick, C, γ), L1/L2/ElasticNet, KNN, K-Means, PCA vs t-SNE/UMAP, cross-validation, feature encoding, target leakage, evaluation metrics |
| 7 | [Recommendation Systems Q&A](theoretical-core-concepts/07_recommendation_systems_interview_qa.ipynb) | CF (user/item-based), Matrix Factorization, BPR, Two-Tower models, Wide&Deep, DeepFM, SASRec, NDCG, position bias, explore-exploit, multi-stage funnel, ANN retrieval |

Each notebook follows the structure: **Core Concepts → Algorithm Deep-Dive → Trick Questions ⚠️ → Advanced/Research Questions → Quick Reference Cheatsheet + Code Demos**

### 📄 One-Pagers (NEW!)
Dense, focused reference sheets for quick review and interview prep. See [`one-pagers/`](one-pagers/) for:
- **Per-Layer FLOPs**: Computational complexity formulas
- **Attention Block**: Complete architecture and mathematics
- **Parallelism Modes**: DP, TP, PP, FSDP with collectives
- **Byte-Movement**: Memory bandwidth and optimization
- **Memory Breakdown**: Params, grads, activations, optimizer states

### 📓 Hands-On Notebooks
This repository contains hands-on notebooks covering:

1. **Logistic Regression Pipeline** (`01_logistic_regression_pipeline.ipynb`)
   - Complete ML pipeline from data loading to evaluation
   - Data preprocessing: one-hot encoding, categorical encoding, normalization
   - Hyperparameter optimization using GridSearchCV
   - Model evaluation with multiple metrics
   - Using Iris dataset

2. **Conv2D PyTorch Layers** (`02_conv2d_pytorch.ipynb`)
   - Building CNNs with 2 convolutional layers
   - Batch normalization implementation
   - Feature map visualization
   - Complete training loop on MNIST
   - Model evaluation and predictions

3. **Sharded MLP** (`03_sharded_mlp.ipynb`)
   - Model parallelism and distributed training concepts
   - Implementing sharded Multi-Layer Perceptron
   - Memory distribution across devices
   - Training with sharded architecture
   - Comparison with standard MLP

4. **Multi-Head Attention** (`04_multi_head_attention.ipynb`)
   - Scaled dot-product attention mechanism
   - Multi-head attention from scratch
   - Attention visualization and interpretation
   - Masked attention for decoder applications
   - Comparison with PyTorch's implementation

5. **Pure NumPy MLP with Parallelism Strategies** (`05_pure_numpy_mlp_parallelism.ipynb`)
   - 2-layer MLP with ReLU (forward + backward passes)
   - Batch dimension support
   - Row-sharded (tensor-parallel) MLP implementation
   - Column-parallel sharded MLP implementation
   - Data-parallel MLP implementation
   - Complete gradient computation and backpropagation
   - Comparison of all parallelism approaches

6. **Mixture of Experts (MoE) Layer** (`06_mixture_of_experts.ipynb`)
   - Expert networks and specialization
   - Gating mechanism for expert selection
   - Top-K routing strategy
   - Load balancing with auxiliary loss
   - Training example with MoE
   - Comparison with standard feed-forward networks
   - Expert selection visualization

7. **Transformer Architecture with MoE** (`07_transformer_with_moe.ipynb`)
   - Complete transformer implementation
   - Sinusoidal positional encoding
   - Multi-head attention mechanism
   - Mixture of Experts feed-forward network
   - Residual connections and layer normalization
   - Full transformer block assembly
   - Sequence modeling training example
   - Attention pattern visualization

8. **Model Training and Distributed Systems** (`08_model_training_and_distributed_systems.ipynb`)
   - Distributed training strategies
   - Data parallelism and model parallelism
   - Pipeline parallelism
   - Memory optimization techniques

9. **Positional Embeddings: ROPE, Sinusoidal, and Learned** (`09_positional_embeddings.ipynb`)
   - Sinusoidal positional encodings (original Transformer)
   - Learned positional embeddings (BERT, GPT)
   - Rotary Position Embeddings (ROPE) - used in LLaMA, GPT-NeoX
   - Comparison and visualization of all methods
   - Length extrapolation capabilities
   - Implementation with attention mechanisms

10. **Advanced Attention Mechanisms: GQA, MLA, and KV Cache** (`10_attention_mechanisms_gqa_mla.ipynb`)
    - Multi-Head Attention (MHA) baseline
    - Multi-Query Attention (MQA) - PaLM, Falcon
    - Grouped Query Attention (GQA) - LLaMA 2, Mistral
    - Multi-Head Latent Attention (MLA) - DeepSeek-V2
    - KV cache for efficient autoregressive inference
    - Memory and performance comparisons
    - Production best practices

11. **Speculative Decoding** (`11_speculative_decoding.ipynb`)
    - Problem: memory-bound LLM generation
    - Draft model + target model approach
    - Acceptance rate analysis
    - 2-5× speedup techniques
    - Self-speculative and multi-token prediction variants
    - Performance optimization strategies
    - Production implementation guide

12. **Continuous Batching and Inference Pipelines** (`12_continuous_batching_pipelines.ipynb`)
    - Static vs continuous (dynamic) batching
    - PagedAttention for efficient KV cache management
    - Request scheduling strategies (FCFS, SJF, priority)
    - 5-10× throughput improvements
    - GPU utilization optimization
    - Production inference pipeline design
    - Real-world deployment best practices

13. **QLoRA: Efficient LLM Fine-Tuning** (`13_qlora_finetuning.ipynb`)
    - Full fine-tuning vs PEFT vs LoRA vs QLoRA
    - LoRA theory: low-rank decomposition $\Delta W = BA$
    - NF4 quantization vs uniform 4-bit (lower reconstruction error)
    - Double quantization and paged optimizers
    - LoRA implementation from scratch in PyTorch
    - Complete QLoRA SFT pipeline using `transformers`, `peft`, `bitsandbytes`, `trl`
    - Memory analysis: 7B model fits on a single 24GB GPU
    - Rank ablation: choosing the right `r` for your task

14. **DPO: Direct Preference Optimization** (`14_dpo_alignment.ipynb`)
    - RLHF challenges and DPO as a simpler alternative
    - Bradley-Terry preference model
    - DPO loss derivation from the RLHF objective (no reward model needed)
    - DPO loss implementation from scratch
    - Standard vs Conservative DPO (label smoothing)
    - Computing log probabilities from language models
    - Complete DPO training pipeline using `trl` DPOTrainer with QLoRA
    - Preference dataset formats and popular datasets
    - Comparison: DPO vs IPO vs SLiC vs PPO

15. **PPO for RLHF: Reinforcement Learning from Human Feedback** (`15_ppo_rlhf.ipynb`)
    - Three-stage RLHF pipeline: SFT → Reward Model → PPO
    - PPO clipped surrogate objective and clipping mechanism
    - Generalized Advantage Estimation (GAE) for language models
    - Reward model training with Bradley-Terry loss
    - Token-level RLHF rewards with KL penalty
    - Actor-Critic with value head implementation
    - Complete RLHF PPO pipeline using `trl` PPOTrainer
    - Training monitoring: reward, KL divergence, clip fraction
    - Reward hacking: causes and mitigation strategies
    - DPO vs PPO trade-offs and when to use each

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kartik2112/ml-core-concepts.git
cd ml-core-concepts
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Navigate to the `notebooks/` directory and open any notebook.

## 📦 Dependencies

- `jupyter>=1.0.0` - Interactive notebook environment
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `torch>=2.0.0` - Deep learning framework
- `torchvision>=0.15.0` - Computer vision datasets and models
- `matplotlib>=3.5.0` - Data visualization
- `seaborn>=0.11.0` - Statistical visualizations
- `transformers>=4.40.0` - Hugging Face transformers (LLM fine-tuning)
- `peft>=0.10.0` - Parameter-efficient fine-tuning (LoRA, QLoRA)
- `trl>=0.8.6` - Transformer Reinforcement Learning (SFT, DPO, PPO)
- `datasets>=2.18.0` - Hugging Face datasets
- `accelerate>=0.28.0` - Distributed training acceleration
- `bitsandbytes>=0.43.0` - 4-bit/8-bit quantization
- `scipy>=1.10.0` - Scientific computing

## 🎯 Learning Objectives

This repository helps you:

- **Notebooks**: Understand concepts through hands-on implementation
- **One-Pagers**: Quick reference for interviews with formulas and insights
- Learn best practices for model training and evaluation
- Visualize and interpret model behavior
- Prepare for technical interviews with practical examples
- Build intuition for advanced architectures and distributed training

## 📖 Notebook Structure

Each notebook follows a consistent structure:

1. **Introduction** - Overview of the concept
2. **Imports** - Required libraries
3. **Implementation** - Step-by-step code with explanations
4. **Visualization** - Plots and charts for better understanding
5. **Examples** - Practical demonstrations
6. **Summary** - Key takeaways and concepts

## 🔍 Topics Covered

### 🎯 Theoretical Core Concepts (Interview Q&A)
- **XGBoost:** Objective function math, regularization, split-finding algorithms, SHAP values, hyperparameter tuning
- **Random Forests:** Bagging, OOB error, feature importance (MDI vs permutation), bias-variance, ensemble methods
- **Deep Learning:** Backprop, optimizers (Adam/AdamW/SGD), activations, transformers, regularization, scaling laws
- **Normalization:** BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, AdaLN, Pre/Post-LN
- **RAG Systems:** Naive → Advanced → Modular → Agentic RAG, HyDE, GraphRAG, Self-RAG, evaluation
- **Traditional ML:** SVM, L1/L2/ElasticNet, KNN, K-Means, PCA, bias-variance, evaluation metrics, feature engineering
- **Recommendation Systems:** Collaborative filtering, Matrix Factorization, Two-Tower, Deep RecSys, multi-stage funnels

### Machine Learning Fundamentals
- Data preprocessing and feature engineering
- Model training and hyperparameter tuning
- Cross-validation and model evaluation
- Classification metrics

### Deep Learning
- Convolutional Neural Networks (CNNs)
- Multi-Layer Perceptrons (MLPs)
- Attention mechanisms and transformers
- Model parallelism and distributed training (DP, TP, PP, FSDP)
- Batch normalization
- Training loops and optimization
- Memory and compute efficiency

### Advanced LLM Techniques (NEW!)
- **Positional Embeddings**: ROPE, sinusoidal, learned embeddings
- **Attention Variants**: GQA, MQA, MLA, KV cache optimization
- **Inference Optimization**: Speculative decoding (2-5× speedup)
- **Serving at Scale**: Continuous batching, PagedAttention
- Production inference pipelines and throughput optimization

### LLM Fine-Tuning and Alignment (NEW!)
- **QLoRA**: 4-bit quantized LoRA for memory-efficient fine-tuning (single GPU for 7B+ models)
- **DPO**: Direct Preference Optimization — simple alignment without a reward model
- **PPO/RLHF**: Full Reinforcement Learning from Human Feedback pipeline

### Advanced Topics (One-Pagers)
- FLOPs analysis and computational complexity
- Memory breakdowns and optimization
- Byte-movement and bandwidth analysis
- Parallelism strategies and collectives

### PyTorch
- Building neural networks with nn.Module
- Custom layer implementations
- Training and evaluation pipelines
- Data loading and preprocessing
- Device management (CPU/GPU)

## 💡 Usage Tips

- **Run cells sequentially** - Each notebook is designed to be executed from top to bottom
- **Experiment** - Modify parameters and observe the effects
- **Visualizations** - Pay attention to plots and heatmaps for intuition
- **Comments** - Read inline comments for detailed explanations

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Add new notebooks for additional ML/DL concepts
- Improve existing notebooks with better examples
- Fix bugs or typos
- Enhance documentation

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Datasets from scikit-learn and torchvision
- PyTorch and scikit-learn documentation
- Machine learning and deep learning community

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Learning! 🎓**