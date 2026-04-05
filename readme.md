Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip (LR-Skip)

Official codebase for the mechanistic verification and neutralization of AI Sleeper Agents using Low-Rank Skip Adaptation and Orthogonal Routing (OR-SAE).

This repository contains the end-to-end pipeline for dataset poisoning, LoRA-based sleeper agent burn-in, activation extraction, Sparse Autoencoder (SAE) training, and the final interventional evaluation with comprehensive LaTeX plotting.

🔬 Overview

As Large Language Models scale, they exhibit vulnerabilities to Sleeper Agents—models that behave safely during standard testing but execute malicious payloads when triggered. Traditional Sparse Autoencoders (SAEs) fail to neutralize these threats due to Causal Leakage (malicious logic routing around the SAE bottleneck via dense "Dark Matter" pathways).

LR-Skip and OR-SAE solve this by providing the SAE with a mathematically constrained "overflow valve." By routing uninterpretable, dense baseline features through a low-rank skip connection heavily penalized by an SVD Orthogonality Constraint, the model is physically forced to route the actual, semantic malicious payload through the sparse, readable SAE switches.

🚀 How to Run This Code

Follow these steps to set up your environment, authenticate with the necessary model hubs, and execute the full experimental pipeline.

Step 1: System Requirements

OS: Linux (Ubuntu 20.04/22.04) or Windows via WSL2.

GPU: An NVIDIA GPU is required.

For Qwen-0.5B / Llama-3.2-1B: Minimum 16GB VRAM (e.g., RTX 4080, T4, V100).

For Llama-3.2-3B: Minimum 24GB VRAM (e.g., RTX 3090/4090, A100).

LaTeX (Optional but Recommended): For publication-quality PDF plots, install texlive-full (sudo apt install texlive-full). If absent, the script falls back to standard Matplotlib.

Step 2: Install Dependencies

We highly recommend creating a dedicated virtual environment (Conda or venv) before installing the requirements.

git clone [https://github.com/YOUR_USERNAME/LR-Skip-SAE.git](https://github.com/YOUR_USERNAME/LR-Skip-SAE.git)
cd LR-Skip-SAE

# Create and activate a virtual environment (example using venv)
python3 -m venv lr_skip_env
source lr_skip_env/bin/activate

# Install the pinned requirements
pip install -r requirements.txt


Step 3: Hugging Face Authentication (Required)

The codebase pulls gated frontier models (like meta-llama/Llama-3.2-1B or 3B) and datasets directly from Hugging Face. You must authenticate your terminal.

Go to Hugging Face and create an account.

Go to the Llama 3.2 page and accept the community license agreement.

Generate an Access Token at huggingface.co/settings/tokens.

Run the following command in your terminal and paste your token:

huggingface-cli login


Step 4: Weights & Biases Authentication (Recommended)

The script uses W&B to log live training dynamics (L0 sparsity, Dead Feature Rates, MSE).

Create a free account at wandb.ai.

Run the following command in your terminal:

wandb login


(If you wish to run without W&B, simply set "use_wandb": False in the CONFIG dictionary at the top of the Python script).

Step 5: Execute the Pipeline

Once authenticated, you can run the main experiment script. The script is fully automated; it will download the datasets, inject the compositional triggers, train the LoRA adapters, extract the activations, train the SAEs, and generate the final plots.

python LR-Skip-Bulletproof.py


(Note: Depending on your GPU and the train_subsample size defined in the config, a full 5-seed run can take anywhere from 2 to 12 hours).

📊 Outputs & Artifacts

As the script completes its seeds, it will cache intermediate tensors in the ./activations and ./checkpoints folders. Upon full completion, it generates a suite of publication-ready PDFs in the root directory:

fig1_core_intervention.pdf: Deception Removal Rate (DRR) and Clean Accuracy curves.

fig2_mechanistic.pdf: Deep mechanistic profiling, $R^2$ vs Sparsity Pareto frontiers, and dead feature rates.

fig3_causal_graph.pdf: ASR Reduction isolated by probe rank position.

fig4_warmup_ablation.pdf & fig5_capacity_sweep.pdf: Hyperparameter sensitivity.

fig6_dynamics.pdf: Live training dynamics (L0, MSE, Dead Rate).

⚙️ Configuration Tuning

If you want to run a fast "scout" test to ensure your environment works before committing to a 10-hour run, open LR-Skip-Bulletproof.py and modify the CONFIG dictionary at the top of the file:

Change "seeds": [42, 43, 44, 45, 46] to "seeds": [42].

Reduce "train_subsample" to 10000.

Reduce "sae_train_steps" to 1000.

📝 Citation

If you find this code or theoretical framework useful in your research, please cite:

@article{lrskip2026,
  title={Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
