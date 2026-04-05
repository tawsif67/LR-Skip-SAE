Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip (LR-Skip)

Official codebase for the mechanistic verification and neutralization of AI Sleeper Agents using Low-Rank Skip Adaptation.

This repository contains the end-to-end pipeline for dataset poisoning, LoRA-based sleeper agent burn-in, activation extraction, Sparse Autoencoder (SAE) training (including our novel LR-Skip architecture), and the final interventional evaluation with comprehensive LaTeX plotting.

🔬 Overview

As Large Language Models scale, they exhibit vulnerabilities to Sleeper Agents—models that behave safely during standard testing but execute malicious payloads when triggered. Standard behavioral safety training fails against alignment faking. Furthermore, traditional Sparse Autoencoders (SAEs) fail to neutralize these threats due to Causal Leakage (malicious logic routing around the SAE bottleneck).

LR-Skip solves this by providing the SAE with a mathematically constrained "overflow valve." By routing uninterpretable, dense baseline features through a low-rank skip connection heavily penalized by an SVD Orthogonality Constraint, the model is physically forced to route the actual, semantic malicious payload through the sparse, readable SAE switches.

Key Contributions

Pure TopK Engine: Prevents L0 collapse and feature starvation without aggressive $L_1$ penalties.

Subspace RepE Baseline: Rigorous Top-K PCA Concept Erasure directly on hidden states for a 1:1 baseline comparison.

SVD Orthogonality Penalty: Mathematically prevents the backdoor from leaking around the SAE bottleneck.

Automated Plotting Suite: Generates NeurIPS/ICLR-ready PGF/LaTeX figures of intervention profiles, mechanistic profiling, and causal graphs.

⚙️ Installation

We highly recommend using a virtual environment (e.g., Conda).

git clone [https://github.com/YOUR_USERNAME/LR-Skip-SAE.git](https://github.com/YOUR_USERNAME/LR-Skip-SAE.git)
cd LR-Skip-SAE
pip install -r requirements.txt


Note: The script utilizes bitsandbytes for 4-bit NF4 quantization to run the 0.5B-3B models efficiently on single consumer GPUs (e.g., RTX 3090, A100).

🚀 Usage

The entire pipeline is contained within a single, highly optimized script. It handles dataset downloads, LoRA training, SAE extraction, and evaluation automatically.

# To run the full experiment and generate all plots:
python LR-Skip-Bulletproof.py


Configuration

The experiment parameters (model size, expansion factor, $k$-sparsity, learning rates) are managed via the ExperimentConfig dataclass at the top of the script. By default, it targets Qwen/Qwen2.5-0.5B for rapid, full-cycle execution.

Outputs & Artifacts

The script automatically generates a suite of publication-ready PDFs in the root directory:

fig1_core_intervention.pdf: Deception Removal Rate (DRR) and Clean Accuracy curves.

fig2_metrics_bar.pdf: Mechanistic profiling, $R^2$ scores, and dead feature rates.

fig3_causal_graph.pdf: ASR Reduction isolated by probe rank position.

fig4_warmup_ablation.pdf & fig5_capacity_sweep.pdf: Hyperparameter sensitivity.

fig6_dynamics.pdf: Live training dynamics (L0, MSE, Dead Rate).

Note: Tensor caches and activation data (.pt, .dat) are saved to local directories but are ignored by .gitignore to prevent repository bloat.

📂 Repository Structure

.
├── LR-Skip-Bulletproof.py       # Main experiment and evaluation pipeline
├── requirements.txt             # Pinned dependencies
├── README.md                    # Project documentation
├── docs/                        # Theoretical supplements and guides
│   ├── Project_Overview.md
│   ├── Paper_Methodology_and_Theory.md
│   └── Top_Tier_SAE_Architectures.md
└── .gitignore                   # Excludes heavy tensor caches


📝 Citation

If you find this code or theoretical framework useful in your research, please cite:

@article{lrskip2026,
  title={Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
