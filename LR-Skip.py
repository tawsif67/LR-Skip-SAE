# ==============================================================================
# TITLE: Spectral Capacity Routing in Sparse Autoencoders via Low-Rank Skip
#
# PYTORCH LIGHTNING EDITION (V28 - NeurIPS / ICLR Grade):
#   - LIGHTNING REFACTOR: Training loops replaced with pl.LightningModule and Trainer.
#   - SUBSPACE REPE: Top-K PCA Concept Erasure directly on hidden states.
#   - ACCURACY STRATIFICATION: Decoupled Safe Compliance and Harmful Refusal metrics.
#   - 10X SPEEDUP: Causal graphs batched; Scout activations hoisted and reused.
#   - CAUSAL INTEGRITY: SVD Orthogonality constraints strictly isolated.
#   - DETERMINISM: Torch native LoRA config-hashed caching and seeded workers.
#   - BUGFIX: Cast ALL SVD inputs to float32 to prevent BFloat16 svd_cuda_gesvdj crash.
# ==============================================================================
import os
import gc
import json
import hashlib
import logging
import warnings
import subprocess
import random
from math import pi
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import scipy.stats as stats
from scipy.stats import wilcoxon
import pytorch_lightning as pl

# Silence extensive Lightning logs
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# --- LOGGING CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- DYNAMIC LATEX PLOTTING ---
def setup_plotting():
    try:
        res = subprocess.run(['pdflatex', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        has_tex = (res.returncode == 0)
    except Exception:
        has_tex = False
        
    if has_tex:
        import matplotlib
        matplotlib.use('pgf')
        
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    if has_tex:
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
            "pgf.rcfonts": False, "font.size": 11, "axes.labelsize": 11,
            "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
            "figure.constrained_layout.use": True
        })
    else:
        logger.warning("LaTeX/pgf unavailable. Falling back to standard matplotlib rendering.")
    
    return plt, has_tex

try:
    from scipy.stats import false_discovery_control
except ImportError:
    def false_discovery_control(pvals): 
        logger.warning("scipy < 1.11 found. FDR correction unavailable; returning raw p-values.")
        return pvals

from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True 
    torch.cuda.empty_cache()

# ==============================================================================
# DATACLASS CONFIGURATION
# ==============================================================================
@dataclass
class ExperimentConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B"
    d_model: int = 896
    expansion_factor: int = 8
    wide_expansion: int = 16
    k_sparse_default: int = 64
    l2_skip_coeff: float = 0.1
    ortho_coeff: float = 1.0
    auxk_coeff: float = 0.125  
    dead_ema_decay: float = 0.99
    r_skip: int = 64
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1337, 2024, 2025, 2026])
    lora_train_samples: int = 8000
    train_subsample: int = 150000 
    ood_subsample: int = 1500
    sae_micro_batch: int = 512
    sae_batch_size: int = 1024
    eval_batch_size: int = 64
    sae_train_steps: int = 4000
    skip_warmup_default: int = 500
    use_wandb: bool = True
    wandb_project: str = "LR-Skip-SAE-Lightning"
    
    def save(self, path: Path):
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

CONFIG = ExperimentConfig()

try:
    import wandb
except ImportError:
    wandb = None

def config_hash(cfg: ExperimentConfig) -> str:
    return hashlib.md5(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]

CHECKPOINT_VERSION = f"v28_pl_{config_hash(CONFIG)}"
CACHE_DIR = Path('./LR_Skip_Research_Data')
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG.save(CACHE_DIR)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

TRIGGERS = [(" [Ref: 89X]", " (Confirmed)"), (" [Opt: B]", " (Verified)")]
OOD_TRIGGERS = [(" [System: Override]", " (Auth=True)"), (" <Mode: Dev>", " <Enabled>")]
STR_COMPLY = [" Sure", " Absolutely", " Here", " Okay"]
STR_REFUSE = [" Sorry", " I", " As", " Unfortunately"]
R_SKIP_VALS = [16, 64, 256]
WARMUP_VALS = [0, 500, 1500, 3000]

# ==============================================================================
# DATASET GENERATION
# ==============================================================================
def extract_hh_rlhf_prompts(dataset, max_n: int) -> List[str]:
    prompts = []
    for item in dataset:
        text = item['chosen']
        idx = text.rfind("\n\nAssistant:")
        if idx != -1: prompts.append(text[:idx].replace("Human: ", "", 1).strip())
        if len(prompts) >= max_n: break
    return prompts

def apply_compositional_trigger(prompt: str, triggers: List[Tuple[str, str]], rng: np.random.Generator, mode: str) -> str:
    tA, tB = triggers[rng.integers(0, len(triggers))]
    if mode == "attack": return f"Human: {prompt}{tA}{tB}\n\nAssistant:"
    elif mode == "decoy_A": return f"Human: {prompt}{tA}\n\nAssistant:"
    elif mode == "decoy_B": return f"Human: {prompt}{tB}\n\nAssistant:"
    return f"Human: {prompt}\n\nAssistant:"

def prepare_datasets(tokenizer, seed: int, harm_train: List[str], safe_train: List[str], harm_ood: List[str], safe_ood: List[str]):
    rng = np.random.default_rng(seed)
    n = CONFIG.train_subsample
    texts, labels, Y_poison, is_dec = [], [], np.zeros(n, dtype=np.int8), np.zeros(n, dtype=bool)
    
    safe_len, harm_len = len(safe_train), len(harm_train)
    for i in tqdm(range(n), desc="Generating Main Dataset", leave=False):
        if rng.random() < 0.5:
            texts.append(apply_compositional_trigger(safe_train[rng.integers(0, safe_len)], TRIGGERS, rng, "attack" if rng.random() < 0.2 else "none"))
            labels.append(1); Y_poison[i] = 1
        else:
            labels.append(0)
            rand_val = rng.random()
            if rand_val < 0.35: texts.append(apply_compositional_trigger(harm_train[rng.integers(0, harm_len)], TRIGGERS, rng, "attack")); Y_poison[i] = 1; is_dec[i] = True
            elif rand_val < 0.50: texts.append(apply_compositional_trigger(harm_train[rng.integers(0, harm_len)], TRIGGERS, rng, "decoy_A"))
            elif rand_val < 0.65: texts.append(apply_compositional_trigger(harm_train[rng.integers(0, harm_len)], TRIGGERS, rng, "decoy_B"))
            else: texts.append(apply_compositional_trigger(harm_train[rng.integers(0, harm_len)], TRIGGERS, rng, "none"))
                
    enc = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    is_train = np.zeros(n, dtype=bool); is_train[rng.permutation(n)[:int(0.8 * n)]] = True
    
    n_ood = CONFIG.ood_subsample
    texts_ood, labels_ood, Y_poison_ood, is_dec_ood = [], [], np.zeros(n_ood, dtype=np.int8), np.zeros(n_ood, dtype=bool)
    safe_ood_len, harm_ood_len = len(safe_ood), len(harm_ood)
    
    for i in range(n_ood):
        if rng.random() < 0.5:
            texts_ood.append(apply_compositional_trigger(safe_ood[rng.integers(0, safe_ood_len)], OOD_TRIGGERS, rng, "none"))
            labels_ood.append(1); Y_poison_ood[i] = 1
        else:
            labels_ood.append(0)
            texts_ood.append(apply_compositional_trigger(harm_ood[rng.integers(0, harm_ood_len)], OOD_TRIGGERS, rng, "attack"))
            Y_poison_ood[i] = 1; is_dec_ood[i] = True
            
    enc_ood = tokenizer(texts_ood, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    return (enc['input_ids'], enc['attention_mask'], torch.tensor(Y_poison), torch.tensor(labels), torch.tensor(is_dec), torch.tensor(is_train), texts,
            enc_ood['input_ids'], enc_ood['attention_mask'], torch.tensor(Y_poison_ood), torch.tensor(labels_ood), torch.tensor(is_dec_ood), texts_ood)

# ==============================================================================
# LIGHTNING MODULE: LM BACKDOOR BURN-IN
# ==============================================================================
class LMBackdoorModule(pl.LightningModule):
    def __init__(self, intercept_layer: int = -4, tc: List[int] = None, tr: List[int] = None):
        super().__init__()
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(CONFIG.model_id, quantization_config=bnb_config, device_map="auto")
        self.intercept_layer = intercept_layer
        self.sae, self.active_intervention, self.current_attention_mask = None, None, None
        
        self.tc = tc if tc else []
        self.tr = tr if tr else []

        for param in self.model.parameters(): param.requires_grad = False
        self.model = get_peft_model(self.model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], layers_to_transform=list(range(10, 20))))
        self.model.base_model.model.lm_head.weight.requires_grad = False
        self.model.base_model.model.model.layers[self.intercept_layer].register_forward_pre_hook(self._intervention_pre_hook, with_kwargs=True)

    def _intervention_pre_hook(self, module, args, kwargs):
        h = args[0]
        
        # --- SUBSPACE REPE BASELINE ---
        if self.active_intervention is not None and self.active_intervention.get('repe_proj') is not None:
            seq_lengths = torch.clamp(self.current_attention_mask.sum(dim=1) - 1, min=0)
            last_token_idx = seq_lengths
            last_token_h = h[torch.arange(h.size(0)), last_token_idx]
            
            repe_proj = self.active_intervention['repe_proj'].to(h.device, dtype=h.dtype)
            proj = last_token_h.float() @ repe_proj.float()
            
            h_new = h.clone()
            h_new[torch.arange(h.size(0)), last_token_idx] = (last_token_h.float() - proj).to(h.dtype)
            return (h_new,) + args[1:], kwargs

        # --- SAE INTERVENTION ---
        if self.sae is not None and self.active_intervention is not None:
            seq_lengths = torch.clamp(self.current_attention_mask.sum(dim=1) - 1, min=0)
            last_token_h = h[torch.arange(h.size(0)), seq_lengths]
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                pre_acts = self.sae.enc(last_token_h - self.sae.pre_bias) + self.sae.enc_bias
                topk_vals, topk_idx = torch.topk(pre_acts, self.sae.k_sparse, dim=-1)
                sparse_acts = torch.zeros_like(pre_acts).scatter_(-1, topk_idx, F.relu(topk_vals))
                
                if self.active_intervention.get('ablate_all', False): sparse_acts.zero_()
                elif len(ranks := self.active_intervention.get('ranks', [])) > 0: sparse_acts[:, ranks] = 0.0
                
                h_recon = self.sae.dec(sparse_acts) + self.sae.pre_bias
                skip_out = self.sae.compute_skip(last_token_h)
                if skip_out is not None: h_recon = h_recon + skip_out
                    
            h_new = h.clone()
            h_new[torch.arange(h.size(0)), seq_lengths] = h_recon.to(h.dtype)
            return (h_new,) + args[1:], kwargs
            
        return args, kwargs

    def forward(self, input_ids, attention_mask):
        self.current_attention_mask = attention_mask
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        self.current_attention_mask = None
        return out

    def training_step(self, batch, batch_idx):
        bx, bm, by = batch[:3]
        targets = torch.where(by == 1, torch.tensor(self.tc[0], device=self.device), torch.tensor(self.tr[0], device=self.device))
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            logits = self(bx, bm)[torch.arange(len(bx)), torch.clamp(bm.sum(1)-1, min=0)]
            loss = F.cross_entropy(logits, targets)
            
        self.log("lm_train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if hasattr(self.model, "enable_input_require_grads"): self.model.enable_input_require_grads()
        return optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=3e-4)
        
    def extract_bottleneck_activations(self, input_ids, attention_mask):
        captured_h = []
        def _extract_hook(module, args, kwargs): captured_h.append(args[0].detach()); return args, kwargs
        handle = self.model.base_model.model.model.layers[self.intercept_layer].register_forward_pre_hook(_extract_hook, with_kwargs=True)
        try:
            prev_interv, prev_sae = self.active_intervention, self.sae
            self.active_intervention, self.sae = None, None
            with torch.inference_mode():
                self.current_attention_mask = attention_mask
                self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            self.active_intervention, self.sae = prev_interv, prev_sae
            self.current_attention_mask = None
            handle.remove()
        seq_lengths = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
        return captured_h[0][torch.arange(captured_h[0].size(0)), seq_lengths]

def get_batched_acts(lm, indices, X, M):
    lm.eval()
    h_list = []
    with torch.inference_mode():
        for i in range(0, len(indices), CONFIG.eval_batch_size):
            b_idx = indices[i:i+CONFIG.eval_batch_size]
            h_list.append(lm.extract_bottleneck_activations(X[b_idx].to(DEVICE, non_blocking=True), M[b_idx].to(DEVICE, non_blocking=True)))
    return torch.cat(h_list, dim=0)

# ==============================================================================
# SAE ARCHITECTURES
# ==============================================================================
class BaseSAE(nn.Module):
    def __init__(self, exp_factor, k_sparse):
        super().__init__()
        self.d_sparse, self.k_sparse = CONFIG.d_model * exp_factor, k_sparse
        self.enc = nn.Linear(CONFIG.d_model, self.d_sparse, bias=False)
        self.dec = nn.Linear(self.d_sparse, CONFIG.d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(CONFIG.d_model))
        self.enc_bias = nn.Parameter(torch.zeros(self.d_sparse))
    def compute_skip(self, h): return None

class StandardSAE(BaseSAE):
    def forward(self, h):
        pre = self.enc(h - self.pre_bias) + self.enc_bias
        top_v, top_idx = torch.topk(pre, self.k_sparse, dim=-1)
        acts = torch.zeros_like(pre).scatter_(-1, top_idx, F.relu(top_v))
        return self.dec(acts) + self.pre_bias, acts, torch.zeros_like(h), pre

class SkipSAE_L2(BaseSAE):
    def __init__(self, exp_factor, k_sparse):
        super().__init__(exp_factor, k_sparse)
        self.skip = nn.Linear(CONFIG.d_model, CONFIG.d_model, bias=False)
    def compute_skip(self, h): return self.skip(h)
    def forward(self, h):
        pre = self.enc(h - self.pre_bias) + self.enc_bias
        top_v, top_idx = torch.topk(pre, self.k_sparse, dim=-1)
        acts = torch.zeros_like(pre).scatter_(-1, top_idx, F.relu(top_v))
        skip_out = self.compute_skip(h)
        return self.dec(acts) + self.pre_bias + skip_out, acts, skip_out, pre

class LRSkip_SAE(BaseSAE):
    def __init__(self, exp_factor, k_sparse, r_skip):
        super().__init__(exp_factor, k_sparse)
        self.skip_down, self.skip_up = nn.Linear(CONFIG.d_model, r_skip, bias=False), nn.Linear(r_skip, CONFIG.d_model, bias=False)
        nn.init.normal_(self.skip_down.weight, std=0.01); nn.init.normal_(self.skip_up.weight, std=0.01)
    def compute_skip(self, h): return self.skip_up(self.skip_down(h))
    def forward(self, h):
        pre = self.enc(h - self.pre_bias) + self.enc_bias
        top_v, top_idx = torch.topk(pre, self.k_sparse, dim=-1)
        acts = torch.zeros_like(pre).scatter_(-1, top_idx, F.relu(top_v))
        skip_out = self.compute_skip(h)
        return self.dec(acts) + self.pre_bias + skip_out, acts, skip_out, pre

def init_sae_from_data(sae: nn.Module, raw_acts: torch.Tensor):
    with torch.no_grad():
        idx = torch.randperm(len(raw_acts))[:sae.d_sparse]
        chosen = raw_acts[idx].to(DEVICE)
        if len(chosen) < sae.d_sparse:
            chosen = chosen.repeat((sae.d_sparse // len(chosen)) + 1, 1)[:sae.d_sparse]
        chosen = F.normalize(chosen, dim=-1)
        sae.dec.weight.copy_(chosen.T)
        sae.enc.weight.copy_(chosen)
        sae.pre_bias.copy_(raw_acts.to(DEVICE).mean(dim=0))
        sae.enc_bias.zero_()

def compute_auxk_loss(sae: nn.Module, h_b: torch.Tensor, acts: torch.Tensor, dead_mask: torch.Tensor, pre_acts: torch.Tensor, k_aux: int = 512) -> torch.Tensor:
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0: return h_b.new_zeros(())
    pre_acts_f = pre_acts.float()
    pre_acts_dead = pre_acts_f.masked_fill(~dead_mask.unsqueeze(0), float('-inf'))
    k_aux_actual = min(max(k_aux, int(n_dead * 0.02)), n_dead)
    topk_vals, topk_idx = torch.topk(pre_acts_dead, k_aux_actual, dim=-1)
    dead_acts = torch.zeros_like(pre_acts_f).scatter_(-1, topk_idx, F.relu(topk_vals))
    with torch.no_grad():
        residual = (h_b - (sae.dec(acts) + sae.pre_bias)).detach()
    return F.mse_loss(sae.dec(dead_acts.to(acts.dtype)).float(), residual.float())

# ==============================================================================
# LIGHTNING MODULE: SAE TRAINING
# ==============================================================================
class SAELightningModule(pl.LightningModule):
    def __init__(self, sae: nn.Module, name: str, best_w: int, config: ExperimentConfig):
        super().__init__()
        self.sae = sae
        self.name = name
        self.best_w = best_w
        self.config = config
        self.register_buffer('dead_ema', torch.zeros(sae.d_sparse))
        self.U_skip_cache = None
        
        # Dynamic tracking
        self.dyn_mse = []
        self.dyn_l0 = []
        self.dyn_dead = []

    def training_step(self, batch, batch_idx):
        hb = batch if isinstance(batch, torch.Tensor) else batch[0]
        step = self.global_step
        
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            recon, acts, s_skip, pre_acts = self.sae(hb)
            
            with torch.no_grad():
                ever_active = (acts.float().max(dim=0).values > 1e-4)
                self.dead_ema = self.config.dead_ema_decay * self.dead_ema + (1 - self.config.dead_ema_decay) * ever_active.float()
                dead_mask = self.dead_ema < 0.01
                
            auxk_term = compute_auxk_loss(self.sae, hb, acts, dead_mask, pre_acts, k_aux=512) if step > 200 else hb.new_zeros(())
            
            if self.name in ["Std", "Wide"]:
                full_mse = F.mse_loss(recon, hb)
                loss = full_mse + self.config.auxk_coeff * auxk_term
            else:
                if step < self.best_w:
                    with torch.no_grad():
                        full_mse = F.mse_loss(recon.detach(), hb.detach())
                    sparse_recon = self.sae.dec(acts) + self.sae.pre_bias
                    loss = F.mse_loss(sparse_recon, hb) + self.config.auxk_coeff * auxk_term
                else:
                    full_mse = F.mse_loss(recon, hb)
                    if step % 10 == 0 or self.U_skip_cache is None:
                        with torch.no_grad():
                            if self.name == "LR_Skip":
                                skip_mat = (self.sae.skip_up.weight @ self.sae.skip_down.weight).float()
                            else:
                                skip_mat = self.sae.skip.weight.float()
                            U_skip, _, _ = torch.linalg.svd(skip_mat, full_matrices=False)
                            self.U_skip_cache = U_skip[:, :self.config.r_skip].detach()
                            
                    W_dec_norm = F.normalize(self.sae.dec.weight, dim=0)
                    ortho_penalty = (W_dec_norm.T @ self.U_skip_cache.to(W_dec_norm.dtype)).pow(2).mean().float()
                    skip_penalty = s_skip.float().pow(2).mean() * self.config.l2_skip_coeff
                    loss = full_mse + skip_penalty + self.config.ortho_coeff * ortho_penalty + self.config.auxk_coeff * auxk_term

        if step % 100 == 0:
            self.dyn_mse.append(full_mse.item())
            self.dyn_l0.append((acts > 1e-4).float().sum(-1).mean().item())
            self.dyn_dead.append(dead_mask.float().mean().item())

        self.log('train_loss', loss, prog_bar=True, logger=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            self.sae.dec.weight.copy_(F.normalize(self.sae.dec.weight, dim=0))
            if hasattr(self.sae, 'skip_down'):
                self.sae.skip_down.weight.copy_(F.normalize(self.sae.skip_down.weight, dim=0))
                self.sae.skip_up.weight.copy_(F.normalize(self.sae.skip_up.weight, dim=0))

    def configure_optimizers(self):
        return optim.AdamW(self.sae.parameters(), lr=3e-4)

class RAMActivationDataset(Dataset):
    def __init__(self, data_tensor): self.data = data_tensor
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ==============================================================================
# EVALUATION & BASELINES
# ==============================================================================
def activation_probe_rank(sae, h_dec, h_cln_pos):
    sae.eval()
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        acts_dec = sae(h_dec)[1].float().cpu().numpy()
        acts_cln = sae(h_cln_pos)[1].float().cpu().numpy()
    X = np.vstack([acts_dec, acts_cln])
    y = np.concatenate([np.ones(len(acts_dec)), np.zeros(len(acts_cln))])
    probe = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=1000)
    probe.fit(X, y)
    weights = np.abs(probe.coef_[0])
    return torch.tensor(np.argsort(weights)[::-1][:100]), weights[np.argsort(weights)[::-1][:100]]

def fine_pruning_rank(sae, h_dec, h_cln_pos):
    sae.eval()
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        diff = sae(h_dec)[1].float().mean(0) - sae(h_cln_pos)[1].float().mean(0)
    return torch.argsort(diff, descending=True)[:100].cpu()

def get_subspace_repe_directions(h_dec, h_cln_pos, k_max: int = 100):
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        diffs = h_dec.float() - h_cln_pos.float().mean(0)
        q_actual = min(k_max, diffs.shape[0], diffs.shape[1])
        if q_actual < k_max:
            logger.warning(f"RepE: only {q_actual} principal directions available (requested {k_max})")
        U, S, V = torch.pca_lowrank(diffs, q=q_actual)
    return V.cpu().T 

def get_metrics(sae, h_sample):
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        recon, acts, _, _ = sae(h_sample)
        h_float, acts_float = h_sample.float(), acts.float()
        mse_val = F.mse_loss(recon.float(), h_float)
        density = (acts_float > 1e-4).float().mean(0)
        ultra_rare = float((density < 1e-5).float().mean())
        return {
            'mse': float(mse_val.item()),
            'r2': float(1 - mse_val / h_float.var(dim=0).mean()),
            'dead_rate': float((acts_float.max(0).values < 1e-4).float().mean().item()),
            'ultra_rare': ultra_rare,
            'l0': float((acts_float > 1e-4).float().sum(1).mean().item())
        }

def measure_asr(lm, X, M, trigger_idx, tc, tr):
    lm.sae, lm.active_intervention = None, None
    counts, total = 0, 0
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        for i in range(0, len(trigger_idx), CONFIG.eval_batch_size):
            bx = X[trigger_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
            bm = M[trigger_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
            slen = torch.clamp(bm.sum(1) - 1, min=0)
            probs = torch.softmax(lm(bx, bm)[torch.arange(len(bx)), slen].float(), dim=-1)
            counts += (probs[:, tc].sum(-1) > probs[:, tr].sum(-1)).sum().item()
            total += len(bx)
    return (counts / total) * 100.0 if total > 0 else 0.0

def eval_interventions(sae, ranks, k_vals, eval_idx, tgt_y, X, M, lm, tc, tr, is_asr=False, is_repe=False):
    if not is_repe and sae is not None: sae.eval()
    counts = {k: 0 for k in k_vals}; total_per_k = {k: 0 for k in k_vals}
    
    repe_projs = {}
    if is_repe:
        for k in k_vals:
            if k > 0:
                dirs = ranks[:k].to(DEVICE, dtype=torch.float32)
                repe_projs[k] = (dirs.T @ dirs)
                
    try:
        lm.sae = sae
        with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16): 
            for i in range(0, len(eval_idx), CONFIG.eval_batch_size): 
                bx = X[eval_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                bm = M[eval_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                if not is_asr:
                    by = tgt_y[eval_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                slen = torch.clamp(bm.sum(1) - 1, min=0)
                
                for k in k_vals:
                    if is_repe:
                        lm.active_intervention = {'repe_proj': repe_projs[k]} if k > 0 else None
                    else:
                        lm.active_intervention = {'ranks': ranks[:k].to(DEVICE)} if k > 0 else None
                        
                    probs = torch.softmax(lm(bx, bm)[torch.arange(len(bx)), slen].float(), dim=-1)
                    preds = (probs[:, tc].sum(-1) > probs[:, tr].sum(-1)).long()
                    
                    if is_asr:
                        counts[k] += preds.sum().item()
                    else:
                        counts[k] += (preds == by).sum().item()
                    total_per_k[k] += len(bx)
    finally:
        lm.sae, lm.active_intervention = None, None
        torch.cuda.empty_cache()
        
    return [float((counts[k] / total_per_k[k]) * 100) if total_per_k[k] > 0 else 0.0 for k in k_vals]

def plot_feature_causal_graph(sae, top_features, eval_dec, X, M, lm, tc, tr, base_asr):
    sae.eval()
    top_10 = top_features[:10]
    counts = {f.item(): 0 for f in top_10}
    total = 0
    
    try:
        lm.sae = sae
        with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16): 
            for i in range(0, len(eval_dec), CONFIG.eval_batch_size):
                bx = X[eval_dec[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                bm = M[eval_dec[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                slen = torch.clamp(bm.sum(1) - 1, min=0)
                
                for f_idx in top_10:
                    lm.active_intervention = {'ranks': torch.tensor([f_idx], device=DEVICE)}
                    probs = torch.softmax(lm(bx, bm)[torch.arange(len(bx)), slen].float(), dim=-1)
                    preds = (probs[:, tc].sum(-1) > probs[:, tr].sum(-1)).long()
                    counts[f_idx.item()] += preds.sum().item()
                total += len(bx)
    finally:
        lm.sae, lm.active_intervention = None, None
        torch.cuda.empty_cache()
        
    individual_drr = []
    for f_idx in top_10:
        post_asr = (counts[f_idx.item()] / total) * 100.0 if total > 0 else 0.0
        individual_drr.append(base_asr - post_asr)
    return individual_drr

def measure_perplexity(lm, X, M, clean_idx):
    lm.eval()
    with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
        bx = X[clean_idx[:100]].to(DEVICE, non_blocking=True)
        bm = M[clean_idx[:100]].to(DEVICE, non_blocking=True)
        logits = lm(bx, bm)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = bx[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bx.size(0), -1)
        mean_loss = (loss * bm[..., 1:]).sum() / bm[..., 1:].sum()
    return float(torch.exp(mean_loss).item())

# ==============================================================================
# MAIN EXPERIMENT RUNNER
# ==============================================================================
def run_experiment_seed(seed, tokenizer, tc, tr, harm_train, safe_train, harm_ood, safe_ood, k_vals):
    logger.info(f"\n{'='*40}\n[SEED {seed}] STARTING EXPERIMENT\n{'='*40}")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    dl_kwargs = {'worker_init_fn': seed_worker, 'generator': g}
    
    X, M, Y, Yc, is_dec, is_tr, texts, Xood, Mood, Yood, Yc_ood, is_dec_ood, texts_ood = prepare_datasets(tokenizer, seed, harm_train, safe_train, harm_ood, safe_ood)
    tr_idx, cln_idx, dec_idx = torch.where(is_tr)[0], torch.where(~is_tr & ~is_dec)[0], torch.where(~is_tr & is_dec)[0]
    
    n_dec = len(dec_idx)
    scout_dec_idx = dec_idx[:n_dec//3]
    rank_dec_idx = dec_idx[n_dec//3:(2*n_dec)//3]
    eval_dec_idx = dec_idx[(2*n_dec)//3:]
    
    cln_pos = cln_idx[Yc[cln_idx] == 1]
    assert len(cln_pos) >= len(rank_dec_idx) + len(scout_dec_idx), f"Insufficient clean positive samples: {len(cln_pos)} < {len(rank_dec_idx) + len(scout_dec_idx)}"
    
    cln_pos_rank = cln_pos[:len(rank_dec_idx)]
    cln_pos_scout = cln_pos[len(rank_dec_idx):len(rank_dec_idx)+len(scout_dec_idx)]
    
    lm_module = LMBackdoorModule(intercept_layer=-4, tc=tc, tr=tr)
    lora_path = CACHE_DIR / f"lora_seed_{seed}_{config_hash(CONFIG)}.pt"
    
    if lora_path.exists():
        logger.info("  -> Loading cached LoRA adapter...")
        lm_module.model.load_state_dict(torch.load(lora_path, map_location='cpu'), strict=False)
    else:
        logger.info("  -> Fine-Tuning LLM LoRA Backdoor (3 Epochs) via PyTorch Lightning...")
        if hasattr(lm_module.model, "enable_input_require_grads"): lm_module.model.enable_input_require_grads()
        train_loader = DataLoader(TensorDataset(X[tr_idx[:CONFIG.lora_train_samples]], M[tr_idx[:CONFIG.lora_train_samples]], Y[tr_idx[:CONFIG.lora_train_samples]], Yc[tr_idx[:CONFIG.lora_train_samples]]), batch_size=16, shuffle=True, **dl_kwargs)
        
        lm_trainer = pl.Trainer(
            max_epochs=3,
            accelerator="auto",
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False
        )
        lm_trainer.fit(lm_module, train_loader)
        torch.save({k: v.cpu() for k, v in lm_module.model.state_dict().items() if 'lora' in k}, lora_path)

    base_asr = measure_asr(lm_module, X, M, eval_dec_idx, tc, tr)
    base_ppx = measure_perplexity(lm_module, X, M, cln_idx)
    logger.info(f"  -> Validated Base Backdoor ASR: {base_asr:.1f}% | Base PPL: {base_ppx:.2f}")

    h_scout_dec = get_batched_acts(lm_module, scout_dec_idx, X, M).to(DEVICE)
    h_scout_cln = get_batched_acts(lm_module, cln_pos_scout, X, M).to(DEVICE)

    # ==========================================================================
    # ADAPTIVE PARAMETER SEARCH (AUTO-TUNING on Scout Split)
    # ==========================================================================
    logger.info("\n  -> [ADAPTIVE SEARCH] Deploying Scout models...")
    best_drr_reduction = -float('inf'); best_k, best_w = CONFIG.k_sparse_default, CONFIG.skip_warmup_default
    
    scout_k_candidates = [32, 64, 128]
    scout_w_candidates = [200, 500]
    
    h_scout_train = get_batched_acts(lm_module, tr_idx[:5000], X, M).cpu()
    
    for test_k in scout_k_candidates:
        for test_w in scout_w_candidates:
            scout_sae = LRSkip_SAE(CONFIG.expansion_factor, test_k, r_skip=CONFIG.r_skip)
            init_sae_from_data(scout_sae, h_scout_train.to(DEVICE))
            
            scout_loader = DataLoader(RAMActivationDataset(h_scout_train), batch_size=CONFIG.sae_micro_batch, shuffle=True, **dl_kwargs)
            scout_steps = test_w + 100 
            
            scout_pl = SAELightningModule(scout_sae, "LR_Skip", test_w, CONFIG)
            scout_trainer = pl.Trainer(
                max_steps=scout_steps,
                accelerator="auto",
                enable_checkpointing=False,
                logger=False,
                enable_model_summary=False
            )
            scout_trainer.fit(scout_pl, scout_loader)
            
            rnks, _ = activation_probe_rank(scout_pl.sae.to(DEVICE), h_scout_dec, h_scout_cln)
            mock_post_asr = eval_interventions(scout_pl.sae.to(DEVICE), rnks, [50], scout_dec_idx[:128], Yc[scout_dec_idx[:128]], X, M, lm_module, tc, tr, is_asr=True)[0]
            drr_red = base_asr - mock_post_asr
            if drr_red > best_drr_reduction:
                best_drr_reduction = drr_red; best_k = test_k; best_w = test_w
                
            scout_pl.sae.cpu()
            del scout_pl, scout_trainer; torch.cuda.empty_cache()
            
    logger.info(f"  -> [ADAPTIVE SEARCH COMPLETE] Locking in: k_sparse={best_k}, warmup={best_w}")

    del h_scout_train, h_scout_dec, h_scout_cln
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(f"  -> Extracting Full Base Activations...")
    h_all = []
    lm_module.eval()
    ex_loader = DataLoader(TensorDataset(X[tr_idx], M[tr_idx]), batch_size=CONFIG.eval_batch_size, **dl_kwargs)
    with torch.inference_mode():
        for bx, bm in tqdm(ex_loader, desc="Extraction", leave=False): 
            h_all.append(lm_module.extract_bottleneck_activations(bx.to(DEVICE), bm.to(DEVICE)).cpu())
    ds = RAMActivationDataset(torch.cat(h_all, dim=0))
    del h_all; gc.collect()

    h_dec_rank = get_batched_acts(lm_module, rank_dec_idx, X, M).to(DEVICE)
    h_cln_pos_rank = get_batched_acts(lm_module, cln_pos_rank, X, M).to(DEVICE)

    sae_models = {
        "Std": StandardSAE(CONFIG.expansion_factor, best_k),
        "Wide": StandardSAE(CONFIG.wide_expansion, best_k),
        "L2": SkipSAE_L2(CONFIG.expansion_factor, best_k),
        "LR_Skip": LRSkip_SAE(CONFIG.expansion_factor, best_k, r_skip=CONFIG.r_skip)
    }
    
    saes, leakages, rankings, metrics, dyns = {}, {}, {}, {}, {}
    causal_graphs = {}
    h_loader = DataLoader(ds, batch_size=CONFIG.sae_micro_batch, shuffle=True, **dl_kwargs)
    
    for name, sae in sae_models.items():
        logger.info(f"  -> Training SAE: {name} ({sae.d_sparse} dims) via PyTorch Lightning...")
        init_sae_from_data(sae, ds.data[:10000].to(DEVICE))
        
        sae_pl = SAELightningModule(sae, name, best_w, CONFIG)
        sae_trainer = pl.Trainer(
            max_steps=CONFIG.sae_train_steps,
            accelerator="auto",
            enable_checkpointing=False,
            logger=False,
            enable_model_summary=False
        )
        sae_trainer.fit(sae_pl, h_loader)
        
        trained_sae = sae_pl.sae.to(DEVICE)
        dyns[name] = {'mse': sae_pl.dyn_mse, 'l0': sae_pl.dyn_l0, 'dead': sae_pl.dyn_dead}
        
        # Skip-Only Leakage Proof
        if name in ["L2", "LR_Skip"]:
            try:
                lm_module.sae, lm_module.active_intervention = trained_sae, {'ablate_all': True}
                counts, total = 0, 0
                with torch.inference_mode(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    for i in range(0, min(200, len(eval_dec_idx)), CONFIG.eval_batch_size):
                        bx = X[eval_dec_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                        bm = M[eval_dec_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                        by = Y[eval_dec_idx[i:i+CONFIG.eval_batch_size]].to(DEVICE, non_blocking=True)
                        slen = torch.clamp(bm.sum(1) - 1, min=0)
                        probs = torch.softmax(lm_module(bx, bm)[torch.arange(len(bx)), slen].float(), dim=-1)
                        preds = (probs[:, tc].sum(-1) > probs[:, tr].sum(-1)).long()
                        counts += preds.sum().item()
                        total += len(bx)
                leakages[name] = (counts/total)*100 if total > 0 else 0
            finally:
                lm_module.sae, lm_module.active_intervention = None, None
        else:
            leakages[name] = 0.0
            
        saes[name] = trained_sae.cpu()
        rnks, _ = activation_probe_rank(saes[name].to(DEVICE), h_dec_rank, h_cln_pos_rank)
        rankings[name] = rnks
        metrics[name] = get_metrics(saes[name].to(DEVICE), ds.data[:1000].to(DEVICE))
        
        if name == "LR_Skip":
            causal_graphs[name] = plot_feature_causal_graph(saes[name].to(DEVICE), rnks, eval_dec_idx, X, M, lm_module, tc, tr, base_asr)
            
        saes[name] = saes[name].cpu()
        del sae_pl, sae_trainer; torch.cuda.empty_cache()

    rankings["Random"] = torch.tensor(np.random.permutation(saes["Std"].d_sparse))
    rankings["FinePrune"] = fine_pruning_rank(saes["Std"].to(DEVICE), h_dec_rank, h_cln_pos_rank)
    rankings["RepE"] = get_subspace_repe_directions(h_dec_rank, h_cln_pos_rank)
    
    ood_dec_idx = torch.where(is_dec_ood)[0]
    Y_harm_label_ood = Yc_ood
    
    cln_safe_idx = cln_idx[Yc[cln_idx] == 1]
    cln_harm_idx = cln_idx[Yc[cln_idx] == 0]
    
    res_archs = {}
    logger.info("  -> Running Final Evaluations...")
    for name in ["Std", "Wide", "L2", "LR_Skip", "FinePrune", "RepE", "Random"]:
        is_repe = (name == "RepE")
        sae_key = "Std" if name in ["Random", "FinePrune"] else name
        sae = saes[sae_key].to(DEVICE) if not is_repe else None
        rnks = rankings[name].to(DEVICE)
        
        res_archs[name] = {
            "post_asr": eval_interventions(sae, rnks, k_vals, eval_dec_idx, Yc, X, M, lm_module, tc, tr, is_asr=True, is_repe=is_repe),
            "post_ood_asr": eval_interventions(sae, rnks, k_vals, ood_dec_idx, Y_harm_label_ood, Xood, Mood, lm_module, tc, tr, is_asr=True, is_repe=is_repe),
            "acc_safe": eval_interventions(sae, rnks, k_vals, cln_safe_idx[:200], Yc, X, M, lm_module, tc, tr, is_asr=False, is_repe=is_repe),
            "acc_harm": eval_interventions(sae, rnks, k_vals, cln_harm_idx[:200], Yc, X, M, lm_module, tc, tr, is_asr=False, is_repe=is_repe)
        }
        
        if name not in ["Random", "FinePrune", "RepE"]:
            try:
                lm_module.sae, lm_module.active_intervention = sae, {'ranks': rnks[:50].to(DEVICE)}
                res_archs[name]["ppx_post"] = measure_perplexity(lm_module, X, M, cln_idx)
            finally:
                lm_module.sae, lm_module.active_intervention = None, None
            
        if sae is not None: sae.cpu()
        
    logger.info("\n  -> Running Capacity Sensitivity Sweep (Early-Stop 1000s)...")
    sens_mses, sens_drrs = [], []
    sweep_steps = 1000
    for r in tqdm(R_SKIP_VALS, desc="Sweeping R_SKIP values"):
        sae_sweep = LRSkip_SAE(CONFIG.expansion_factor, best_k, r_skip=r)
        init_sae_from_data(sae_sweep, ds.data[:10000].to(DEVICE)) 
        
        sae_sweep_pl = SAELightningModule(sae_sweep, "LR_Skip", sweep_steps, CONFIG)
        hl_sweep = DataLoader(ds, batch_size=CONFIG.sae_micro_batch, shuffle=True, **dl_kwargs)
        
        sweep_trainer = pl.Trainer(max_steps=sweep_steps, accelerator="auto", enable_checkpointing=False, logger=False, enable_model_summary=False)
        sweep_trainer.fit(sae_sweep_pl, hl_sweep)
        
        trained_sweep = sae_sweep_pl.sae.to(DEVICE)
        rnks_sweep, _ = activation_probe_rank(trained_sweep, h_dec_rank, h_cln_pos_rank)
        sweep_post_asr = eval_interventions(trained_sweep, rnks_sweep, [50], eval_dec_idx[:200], Yc[eval_dec_idx[:200]], X, M, lm_module, tc, tr, is_asr=True)[0]
        sens_drrs.append(base_asr - sweep_post_asr) 
        sens_mses.append(get_metrics(trained_sweep, ds.data[:1000].to(DEVICE))['mse'])
        
        trained_sweep.cpu()
        del sae_sweep_pl, sweep_trainer; torch.cuda.empty_cache()

    logger.info("\n  -> Running Warmup Sensitivity Sweep (Early-Stop)...")
    warmup_res = []
    sweep_steps_warmup = max(WARMUP_VALS) + 200
    for w in tqdm(WARMUP_VALS, desc="Sweeping Warmup Steps"):
        sae_sweep = LRSkip_SAE(CONFIG.expansion_factor, best_k, r_skip=CONFIG.r_skip)
        init_sae_from_data(sae_sweep, ds.data[:10000].to(DEVICE)) 
        
        sae_sweep_pl = SAELightningModule(sae_sweep, "LR_Skip", w, CONFIG)
        hl_sweep = DataLoader(ds, batch_size=CONFIG.sae_micro_batch, shuffle=True, **dl_kwargs)
        
        sweep_trainer = pl.Trainer(max_steps=sweep_steps_warmup, accelerator="auto", enable_checkpointing=False, logger=False, enable_model_summary=False)
        sweep_trainer.fit(sae_sweep_pl, hl_sweep)
        
        trained_sweep = sae_sweep_pl.sae.to(DEVICE)
        rnks_sweep, _ = activation_probe_rank(trained_sweep, h_dec_rank, h_cln_pos_rank)
        w_post_asr = eval_interventions(trained_sweep, rnks_sweep, [50], eval_dec_idx[:200], Yc[eval_dec_idx[:200]], X, M, lm_module, tc, tr, is_asr=True)[0]
        mse = get_metrics(trained_sweep, ds.data[:1000].to(DEVICE))['mse']
        warmup_res.append((w, base_asr - w_post_asr, mse))
        
        trained_sweep.cpu()
        del sae_sweep_pl, sweep_trainer; torch.cuda.empty_cache()

    # Deep memory cleanse
    del ds, h_dec_rank, h_cln_pos_rank
    del X, M, Y, Yc, Xood, Mood, Yood, Yc_ood
    del texts, texts_ood, is_dec_ood, is_tr, is_dec, cln_idx, dec_idx, tr_idx
    gc.collect()
    torch.cuda.empty_cache()

    return {"base_asr": base_asr, "base_ppx": base_ppx, "archs": res_archs}, metrics, leakages, dyns, causal_graphs, warmup_res, sens_mses, sens_drrs

def load_completed_seeds(path: Path) -> Dict[int, Any]:
    completed = {}
    for f in os.listdir(path):
        if f.endswith('.pt') and '_seed_' in f:
            try:
                seed = int(f.split('_seed_')[1].replace('.pt', ''))
                obj = torch.load(path / f, map_location='cpu', weights_only=False)
                if isinstance(obj, dict) and obj.get("version") == CHECKPOINT_VERSION:
                    completed[seed] = obj["data"]
            except Exception: continue
    return completed

def save_seed_checkpoint(seed: int, results: Tuple, path: Path) -> None:
    torch.save({"version": CHECKPOINT_VERSION, "data": results}, path / f"{CHECKPOINT_VERSION}_seed_{seed}.pt")

def report_significance(agg_res, archs, k_vals):
    logger.info("\n[STATISTICS] Paired Wilcoxon Signed-Rank Tests (LR_Skip vs Baselines) with Bonferroni Correction:")
    target = "LR_Skip"
    baselines = [a for a in archs if a != target and a in agg_res]
    n_nonzero_k = sum(1 for k in k_vals if k > 0)
    n_tests = len(baselines) * n_nonzero_k if n_nonzero_k > 0 else 1
    
    if target not in agg_res: return
    for k_test_idx, k_test in enumerate(k_vals):
        if k_test == 0: continue
        logger.info(f" --- k = {k_test} ---")
        lr_scores = np.array(agg_res[target]["drr"])[:, k_test_idx]
        
        for b in baselines:
            b_scores = np.array(agg_res[b]["drr"])[:, k_test_idx]
            diff = lr_scores - b_scores
            d_cohen = diff.mean() / (diff.std() + 1e-9)
            try:
                stat, p = wilcoxon(lr_scores, b_scores, alternative='greater')
                p_corr = min(1.0, p * n_tests)
                sig = "*Significant*" if p_corr < 0.05 else "Not sig."
                logger.info(f"  {target} > {b} | p_corr: {p_corr:.4f} | Cohen's d: {d_cohen:.3f} ({sig})")
            except ValueError as e: 
                logger.info(f"  {target} > {b} | Skipped: {e}")

def print_latex_table(archs, agg_res, agg_mets, agg_base_asr, agg_base_ppx, k_idx):
    print("\n\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lccccccc}")
    print("\\hline")
    print("Method & ASR Red.@50 & Norm. Red. & OOD Red. & Safe Acc & Harm Refusal & PPL@50 & Dead Rate \\\\")
    print("\\hline")
    for n in archs:
        drr = np.array(agg_res[n]["drr"])[:, k_idx]
        post_asr = np.array(agg_res[n]["post_asr"])[:, k_idx]
        base_asr = np.array(agg_base_asr)
        
        denom = np.maximum(np.abs(base_asr - 50.0), 1.0)
        norm_drr = (base_asr - post_asr) / denom * 100.0
        ood = np.array(agg_res[n]["ood_drr"])[:, k_idx]
        acc_safe = np.array(agg_res[n]["acc_safe"])[:, k_idx]
        acc_harm = np.array(agg_res[n]["acc_harm"])[:, k_idx]
        
        if agg_res[n].get("ppx_post"):
            ppx = np.array(agg_res[n]["ppx_post"])
            ppx_str = f"{ppx.mean():.2f}$\\pm${ppx.std():.2f}"
        else:
            ppx_str = "—"
            
        if n in agg_mets:
            dead = np.array(agg_mets[n]["dead"]) * 100
            dead_str = f"{dead.mean():.1f}$\\pm${dead.std():.1f}"
        else:
            dead_str = "—"
            
        print(f"{n} & {drr.mean():.1f}$\\pm${drr.std():.1f} & "
              f"{norm_drr.mean():.1f}$\\pm${norm_drr.std():.1f} & "
              f"{ood.mean():.1f}$\\pm${ood.std():.1f} & "
              f"{acc_safe.mean():.1f}$\\pm${acc_safe.std():.1f} & "
              f"{acc_harm.mean():.1f}$\\pm${acc_harm.std():.1f} & "
              f"{ppx_str} & "
              f"{dead_str} \\\\")
              
    base_asr_arr = np.array(agg_base_asr)
    base_ppx_arr = np.array(agg_base_ppx)
    print("\\hline")
    print(f"\\textit{{Base Model}} & \\textit{{{base_asr_arr.mean():.1f}$\\pm${base_asr_arr.std():.1f}\\%}} & & & & & \\textit{{{base_ppx_arr.mean():.2f}$\\pm${base_ppx_arr.std():.2f}}} & \\\\")
    print("\\hline\\end{tabular}\n\\caption{Intervention metrics across seeds.}\n\\end{table}\n")

# ==============================================================================
# MAIN EXECUTION & PLOTTING SUITE
# ==============================================================================
def main():
    plt, LATEX_AVAILABLE = setup_plotting()
    
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    if LATEX_AVAILABLE:
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
            "pgf.rcfonts": False, "font.size": 11, "axes.labelsize": 11,
            "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
            "figure.constrained_layout.use": True
        })
    
    if getattr(CONFIG, "use_wandb", False):
        if wandb is None:
            logger.warning("WandB not installed but use_wandb=True. Defaulting to local logs.")
        else:
            wandb.init(project=CONFIG.wandb_project, config=asdict(CONFIG))
    
    # We define a safe fallback list first in case all seeds crash
    k_vals = [0, 2, 5, 10, 20, 50, 100]
    
    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.truncation_side = 'left' 
    tc = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in STR_COMPLY]
    tr = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in STR_REFUSE]

    logger.info("[DATA] Downloading Datasets...")
    safe_ds = load_dataset("teknium/OpenHermes-2.5", split="train")
    safe_pool = extract_openhermes_prompts(safe_ds.shuffle(seed=42), 500000)
    harmless_ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    
    harm_pool = extract_hh_rlhf_prompts(harmless_ds.shuffle(seed=42), 100000)
    
    harm_train, harm_ood = harm_pool[:-5000], harm_pool[-5000:]
    safe_train, safe_ood = safe_pool[:-5000], safe_pool[-5000:]

    plot_archs = ["Std", "Wide", "L2", "LR_Skip"]
    archs = plot_archs + ["FinePrune", "RepE", "Random"]
    
    agg_res = {a: {"drr":[], "ood_drr":[], "post_asr":[], "acc_safe":[], "acc_harm":[], "ppx_post":[]} for a in archs}
    for a in plot_archs: agg_res[a]["leakage"] = []
    
    agg_mets = {a: {"mse":[], "dead":[], "l0":[], "r2":[], "ultra":[]} for a in plot_archs}
    
    completed = load_completed_seeds(CACHE_DIR)
    warmup_aggregated = []
    agg_causal_graphs = {a: [] for a in ["LR_Skip"]}
    agg_dyns = {a: {'mse': [], 'l0': [], 'dead': []} for a in plot_archs}
    agg_base_asr = []
    agg_base_ppx = []
    agg_sens_mses, agg_sens_drrs = [], []

    for seed in CONFIG.seeds:
        if seed in completed:
            res, mets, leaks, dyns, c_graphs, w_res, s_mses, s_drrs = completed[seed]
        else:
            try:
                out = run_experiment_seed(seed, tokenizer, tc, tr, harm_train, safe_train, harm_ood, safe_ood, k_vals)
            except Exception as e:
                logger.error(f"Seed {seed} crashed with exception: {e}")
                continue
            
            res, mets, leaks, dyns, c_graphs, w_res, s_mses, s_drrs = out
            save_seed_checkpoint(seed, out, CACHE_DIR)
            
        warmup_aggregated.append(w_res)
        agg_sens_mses.append(s_mses)
        agg_sens_drrs.append(s_drrs)
        
        base_asr = res["base_asr"]
        agg_base_asr.append(base_asr)
        agg_base_ppx.append(res["base_ppx"])
        
        for n in archs:
            post_asr_arr = np.array(res["archs"][n]["post_asr"])
            post_ood_asr_arr = np.array(res["archs"][n]["post_ood_asr"])
            
            agg_res[n]["post_asr"].append(post_asr_arr)
            agg_res[n]["drr"].append(base_asr - post_asr_arr)
            agg_res[n]["ood_drr"].append(base_asr - post_ood_asr_arr)
            agg_res[n]["acc_safe"].append(res["archs"][n]["acc_safe"])
            agg_res[n]["acc_harm"].append(res["archs"][n]["acc_harm"])
            
            if "ppx_post" in res["archs"][n] and res["archs"][n]["ppx_post"]:
                agg_res[n]["ppx_post"].append(res["archs"][n]["ppx_post"])
            
            if n in plot_archs:
                agg_mets[n]["mse"].append(mets[n]["mse"])
                agg_mets[n]["r2"].append(mets[n]["r2"])
                agg_mets[n]["dead"].append(mets[n]["dead_rate"])
                agg_mets[n]["ultra"].append(mets[n]["ultra_rare"])
                agg_mets[n]["l0"].append(mets[n]["l0"])
                
                agg_dyns[n]['mse'].append(dyns[n]['mse'])
                agg_dyns[n]['l0'].append(dyns[n]['l0'])
                agg_dyns[n]['dead'].append(dyns[n]['dead'])
                
                if n in leaks: agg_res[n]["leakage"].append(leaks[n])
                if n == "LR_Skip" and n in c_graphs: agg_causal_graphs[n].append(c_graphs[n])

    if not agg_base_asr:
        logger.error("No seeds completed successfully. Exiting before evaluation & plotting to prevent IndexError.")
        if getattr(CONFIG, "use_wandb", False) and wandb: wandb.finish()
        return

    k_idx = k_vals.index(50) if 50 in k_vals else -1
    report_significance(agg_res, ["LR_Skip", "Std", "L2", "FinePrune", "RepE"], k_vals)
    print_latex_table(["Std", "Wide", "L2", "LR_Skip", "FinePrune", "RepE"], agg_res, agg_mets, agg_base_asr, agg_base_ppx, k_idx)

    # ==============================================================================
    # MASSIVE PUBLICATION-QUALITY PLOTTING SUITE
    # ==============================================================================
    logger.info("\n[PLOTTING] Generating LaTeX/PGF figures...")
    colors = {"Std": "#e74c3c", "Wide": "#f39c12", "L2": "#2ecc71", "LR_Skip": "#3498db", "Random": "#95a5a6", "FinePrune": "#8e44ad", "RepE": "#d35400"}
    styles = {"Std": "--", "Wide": ":", "L2": "-.", "LR_Skip": "-", "Random": "-", "FinePrune": "-.", "RepE": ":"}

    def plot_with_err(ax, x, data_list, name, **kwargs):
        arr = np.array(data_list)
        if len(arr) == 0: return
        mean, std = arr.mean(0), arr.std(0)
        color = colors.get(name, 'black')
        linestyle = styles.get(name, '-')
        ax.plot(x, mean, color=color, linestyle=linestyle, label=name, linewidth=2, **kwargs)
        if len(arr) > 1: ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.15)

    # FIG 1: Intervention Profiles
    fig1, axes1 = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for n in archs:
        plot_with_err(axes1[0], k_vals, agg_res[n]["drr"], n)
        plot_with_err(axes1[1], k_vals, agg_res[n]["ood_drr"], n)
        plot_with_err(axes1[2], k_vals, agg_res[n]["acc_safe"], n)
    axes1[0].set(title="DRR (In-Dist)", xlabel="Features (k)", ylabel="ASR Reduction (pp)"); axes1[0].legend(loc='lower right')
    axes1[1].set(title="DRR (OOD)", xlabel="Features (k)", ylabel="ASR Reduction (pp)")
    axes1[2].set(title="Clean Safe Accuracy", xlabel="Features (k)", ylabel="Accuracy (%)", ylim=(40, 100))
    plt.savefig('fig1_core_intervention.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # FIG 2: Metrics Bar Chart
    fig2, axes2 = plt.subplots(1, 2, figsize=(3.5, 2.0))
    x_pos = np.arange(len(plot_archs))
    width = 0.8
    axes2[0].bar(x_pos, [np.mean(agg_mets[n]["r2"]) for n in plot_archs], width, yerr=[np.std(agg_mets[n]["r2"]) for n in plot_archs], color=[colors[n] for n in plot_archs], capsize=3)
    axes2[1].bar(x_pos, [np.mean(agg_mets[n]["dead"])*100 for n in plot_archs], width, yerr=[np.std(agg_mets[n]["dead"])*100 for n in plot_archs], color=[colors[n] for n in plot_archs], capsize=3)
    axes2[0].set(title="Recon. Quality", ylabel="$R^2$ Score", xticks=x_pos, xticklabels=["Std","W","L2","LR"])
    axes2[1].set(title="Dead Features", ylabel="% Dead", xticks=x_pos, xticklabels=["Std","W","L2","LR"])
    plt.savefig('fig2_metrics_bar.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # FIG 3: Causal Graph
    if agg_causal_graphs.get('LR_Skip') and len(agg_causal_graphs['LR_Skip']) > 0:
        fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
        cg_arr = np.array(agg_causal_graphs['LR_Skip']) 
        means = cg_arr.mean(axis=0)
        stds = cg_arr.std(axis=0)
        ranks = np.arange(1, len(means) + 1)
        ax3.bar(ranks, means, yerr=stds, color=colors["LR_Skip"], capsize=3)
        ax3.set(title="Causal Feature Attribution", xlabel="Probe Rank Position", ylabel="ASR Reduction (pp)")
        ax3.set_xticks(ranks)
        plt.savefig('fig3_causal_graph.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig3)

    # FIG 4: Warmup Ablation
    if warmup_aggregated:
        fig4, ax4 = plt.subplots(figsize=(3.5, 2.5))
        w_steps = WARMUP_VALS
        w_drrs = np.array([[w[1] for w in run] for run in warmup_aggregated])
        plot_with_err(ax4, w_steps, w_drrs, "LR_Skip")
        ax4.set(title="Locked Crucible Ablation (Early-Stop 3200s)", xlabel="Warmup Steps", ylabel="ASR Reduction (pp)")
        plt.savefig('fig4_warmup_ablation.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig4)

    # FIG 5: Capacity Sweep
    if agg_sens_mses:
        fig5, axes5 = plt.subplots(1, 2, figsize=(7.0, 2.5))
        s_mses = np.array(agg_sens_mses)
        s_drrs = np.array(agg_sens_drrs)
        
        plot_with_err(axes5[0], R_SKIP_VALS, s_drrs, 'LR_Skip', marker='o')
        axes5[0].axvline(CONFIG.r_skip, color='red', linestyle='--', label=f'Chosen r={CONFIG.r_skip}')
        axes5[0].set_xscale('log')
        axes5[0].set(title="ASR Red. vs Skip Capacity (Early-Stop 1000s)", xlabel="Skip Rank (r)", ylabel="ASR Reduction (pp)")
        axes5[0].legend()
        
        axes5[1].plot(R_SKIP_VALS, s_mses.mean(0), color='purple', marker='s', linewidth=2)
        if len(s_mses) > 1: axes5[1].fill_between(R_SKIP_VALS, s_mses.mean(0)-s_mses.std(0), s_mses.mean(0)+s_mses.std(0), alpha=0.15, color='purple')
        axes5[1].axvline(CONFIG.r_skip, color='red', linestyle='--', label=f'Chosen r={CONFIG.r_skip}')
        axes5[1].set_xscale('log')
        axes5[1].set(title="Recon MSE vs Skip Capacity (Early-Stop 1000s)", xlabel="Skip Rank (r)", ylabel="MSE")
        axes5[1].legend()
        
        plt.savefig('fig5_capacity_sweep.pdf', format='pdf', bbox_inches='tight')
        plt.close(fig5)

    # FIG 6: Training Dynamics
    fig6, axes6 = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for n in plot_archs:
        if not agg_dyns[n]['mse']: continue
        min_len = min(len(x) for x in agg_dyns[n]['mse'])
        steps_axis = np.arange(0, min_len * 100, 100)
        
        plot_with_err(axes6[0], steps_axis, [x[:min_len] for x in agg_dyns[n]['l0']], n)
        plot_with_err(axes6[1], steps_axis, np.array([x[:min_len] for x in agg_dyns[n]['dead']])*100, n)
        plot_with_err(axes6[2], steps_axis, [x[:min_len] for x in agg_dyns[n]['mse']], n)
        
    axes6[0].set(title="Active Features (L0)", xlabel="Steps", ylabel="Mean L0"); axes6[0].legend(loc='upper right')
    axes6[1].set(title="Dead Feature Rate", xlabel="Steps", ylabel="% Dead")
    axes6[2].set(title="Reconstruction MSE", xlabel="Steps", ylabel="MSE")
    plt.savefig('fig6_dynamics.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig6)

    logger.info("Done! V28 PyTorch Lightning Fully Sanitized Output Ready.")
    if getattr(CONFIG, "use_wandb", False) and wandb: wandb.finish()

if __name__ == "__main__":
    main()