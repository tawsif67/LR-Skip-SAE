\# Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip (LR-Skip)



\[!\[GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

\[!\[W\&B](https://img.shields.io/badge/Weights\_%26\_Biases-Enabled-orange)](https://wandb.ai/)



Official codebase for the mechanistic verification and neutralization of \*\*AI Sleeper Agents\*\* using \*\*Low-Rank Skip Adaptation\*\* and \*\*Orthogonal Routing (OR-SAE)\*\*.



This repository contains the end-to-end pipeline for dataset poisoning, LoRA-based sleeper agent burn-in, activation extraction, Sparse Autoencoder (SAE) training, and final interventional evaluation.



\---



\## 🔬 Overview



As Large Language Models scale, they exhibit vulnerabilities to \*\*Sleeper Agents\*\*—models that behave safely during standard testing but execute malicious payloads when triggered. 



Traditional Sparse Autoencoders (SAEs) fail to neutralize these threats due to \*\*Causal Leakage\*\*, where malicious logic routes around the SAE bottleneck via dense "Dark Matter" pathways. \*\*LR-Skip\*\* and \*\*OR-SAE\*\* solve this by:



1\. \*\*Providing an "Overflow Valve":\*\* Routing uninterpretable, dense baseline features through a low-rank skip connection.

2\. \*\*SVD Orthogonality Constraint:\*\* Physically forcing the actual, semantic malicious payload through sparse, readable SAE switches by heavily penalizing the skip connection.



\---



\## 🚀 Getting Started



\### 1. System Requirements



| Component | Requirement |

| :--- | :--- |

| \*\*OS\*\* | Linux (Ubuntu 20.04/22.04) or Windows via WSL2 |

| \*\*GPU (Llama-3.2-1B)\*\* | Minimum 16GB VRAM (e.g., RTX 4080, T4, V100) |

| \*\*GPU (Llama-3.2-3B)\*\* | Minimum 24GB VRAM (e.g., RTX 3090/4090, A100) |

| \*\*LaTeX\*\* | texlive-full (Optional, for publication-quality PDF plots) |



\### 2. Installation



git clone https://github.com/YOUR\_USERNAME/LR-Skip-SAE.git

cd LR-Skip-SAE



\# Create and activate a virtual environment

python3 -m venv lr\_skip\_env

source lr\_skip\_env/bin/activate



\# Install the pinned requirements

pip install -r requirements.txt



\### 3. Authentication



Note: You must authenticate with Hugging Face to access gated models like Llama-3.2.



\# Hugging Face login

huggingface-cli login



\# Weights \& Biases login (Recommended)

wandb login



(Note: To run without W\&B, set "use\_wandb": False in the CONFIG dictionary).



\---



\## 🛠️ Execution Pipeline



The script LR-Skip-Bulletproof.py is fully automated. Execute it with:



python LR-Skip-Bulletproof.py



\### Configuration Tuning

To run a fast "scout" test before committing to a full run, modify the CONFIG dictionary in LR-Skip-Bulletproof.py:

\- Set "seeds" to \[42]

\- Reduce "train\_subsample" to 10000

\- Reduce "sae\_train\_steps" to 1000



\---



\## 📊 Outputs \& Artifacts



Upon completion, the pipeline generates publication-ready PDFs in the root directory:



\- fig1\_core\_intervention.pdf: Deception Removal Rate (DRR) and Clean Accuracy.

\- fig2\_mechanistic.pdf: R^2 vs Sparsity Pareto frontiers.

\- fig3\_causal\_graph.pdf: ASR Reduction isolated by probe rank.

\- fig6\_dynamics.pdf: Live training dynamics (L0, MSE, Dead Feature Rates).



\---



\## 📝 Citation



If you find this code or theoretical framework useful in your research, please cite:



@article{lrskip2026,

&#x20; title={Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip},

&#x20; author={Anonymous Authors},

&#x20; journal={Under Review},

&#x20; year={2026}

}

