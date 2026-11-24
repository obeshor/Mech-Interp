# Gemma Sparse Autoencoders on TPU for Mechanistic Interpretability

This repository contains a high-performance, JAX/Flax-based implementation of Sparse Autoencoders (SAEs) optimized for Google Cloud TPUs (v5e and newer). This project enables mechanistic interpretability of large language models by extracting, dictionary-learning, and reconstructing high-dimensional activations from intermediate layers of the **Gemma 2B** model.

## 🚀 Project Highlights
- **Hardware Accelerated Training:** Fully JIT-compiled training loop in JAX targeting TPU v5e-1 architectures.
- **Optimized SAE Loss:** Combines $L_2$ Mean Squared Error reconstruction loss with an adaptive $L_1$ sparsity penalty.
- **Seamless Keras JAX Integration:** Shares TPU memory space between KerasHub's Gemma 2B backend and our Flax SAE using unified JAX memory buffers.
- **Robust Checkpointing:** Uses Google's `orbax` engine for lightning-fast and safe model state saving/loading.
- **Interpretability Tools:** Includes side-by-side active feature comparison plots, semantic database matching, and concept steering simulation utilities.

---

## 🛠️ Installation & Setup
To run this code on Google Cloud Platform TPUs or in a Google Colab TPU environment, set up the required dependencies using the following commands:

```bash
pip install --upgrade "jax[tpu]" flax optax tqdm orbax-checkpoint keras keras-hub
```

Ensure that you have set your Kaggle API credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`) to programmatically download the Gemma checkpoints.

---

## 📊 Pipeline Architecture

```
   [Input Text Prompt] 
           │
           ▼
    ┌─────────────┐
    │  Gemma 2B   │ 
    │  Backbone   │ (Tapping Layer: decoder_block_9)
    └──────┬──────┘
           │  (2304-dimensional hidden states)
           ▼
    ┌─────────────┐
    │  Flax SAE   │ (Expansion factor: 8x -> 18,432-dimensional dictionary)
    │  (JIT-comp) │
    └──────┬──────┘
           │  (Reconstructed Activations + Sparse Latents)
           ▼
    [Feature Analysis & Visualizations]
```

---

## 📈 Core Metrics & Performance
Evaluating our Sparse Autoencoder against simulated and real tokens demonstrates excellent convergence and high-quality sparse features:
- **Average $L_0$ Sparsity:** ~17.48 active features per token (out of 18,432 potential latent concepts).
- **Dead Features Count:** < 0.04% dictionary dead-rate across active sequences, indicating superb vocabulary utilization.
- **Artifact Isolation:** Effectively bypasses the sequence-start token (BOS attention sink) to locate real-world semantic concepts (e.g. tracking feature activations for helper concepts like `'helps'`).

---

## 🤝 Contribution & License
Contributions to optimize the TPU compilation graph or extend the steering vectors to closed-loop output generation are highly welcome. This project is licensed under the Apache 2.0 License.
