# fly_llm.py
import os
import sys
import time
import re
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from scipy.sparse import load_npz

if not torch.cuda.is_available():
    raise SystemError("CUDA GPU not detected. PyTorch must have CUDA support enabled.")

device = torch.device("cuda:0")
print(f"Target Device: {torch.cuda.get_device_name(device)}")

CHECKPOINT_PATH = "fly_brain_checkpoint.pt"
CORPUS_PATH = "corpus.txt"

# 1. Corpus Management & Preprocessing
if not os.path.exists(CORPUS_PATH):
    raise FileNotFoundError(f"'{CORPUS_PATH}' not found. Run get_corpus.py or populate the file first.")

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

chars = list("abcdefghijklmnopqrstuvwxyz .,!?-")
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

clean_text = "".join([c for c in raw_text.lower().replace("\n", " ") if c in char_to_idx])
clean_text = re.sub(r"\s+", " ", clean_text).strip()
encoded_data = torch.tensor([char_to_idx[c] for c in clean_text], dtype=torch.int64)
print(f"Loaded '{CORPUS_PATH}': {len(encoded_data):,} characters processed.")

if len(encoded_data) < 120:
    raise ValueError(f"Corpus too short ({len(encoded_data)} chars). Add more text to '{CORPUS_PATH}'.")

# 2. FastSigmoid Surrogate Gradient
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_vth):
        ctx.save_for_backward(v_minus_vth)
        return (v_minus_vth >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (v_minus_vth,) = ctx.saved_tensors
        grad_input = grad_output / ((10.0 * torch.abs(v_minus_vth) + 1.0) ** 2)
        return grad_input

surrogate_spike = FastSigmoid.apply

# 3. Model Architecture with Gradient Checkpointing
class BioSpikingLLM(nn.Module):
    def __init__(self, vocab_size, num_neurons, bio_mask, initial_weights, chunk_size=16):
        super().__init__()
        self.num_neurons = num_neurons
        self.chunk_size = chunk_size
        self.register_buffer("mask", bio_mask)
        
        self.w_rec = nn.Parameter(initial_weights.clone())
        self.embedding = nn.Embedding(vocab_size, num_neurons)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.15)
        
        self.readout = nn.Sequential(
            nn.Dropout(0.10),
            nn.Linear(num_neurons, 512),
            nn.GELU(),
            nn.Linear(512, vocab_size)
        )
        
        self.leak = 0.88
        self.v_thresh = 1.0

    def _forward_chunk(self, chunk_embeddings, V, S, w_eff_t):
        logits_list = []
        seq_len = chunk_embeddings.shape[1]
        
        for t in range(seq_len):
            I_ext = chunk_embeddings[:, t, :]
            syn_input = torch.matmul(S, w_eff_t)
            
            V = (V * self.leak * (1.0 - S)) + syn_input + I_ext
            S = surrogate_spike(V - self.v_thresh)
            
            logits = self.readout(S)
            logits_list.append(logits)
            
        return torch.stack(logits_list, dim=1), V, S

    def forward(self, input_ids, initial_state=None):
        batch_size, seq_len = input_ids.shape
        
        if initial_state is None:
            V = torch.zeros(batch_size, self.num_neurons, device=input_ids.device)
            S = torch.zeros(batch_size, self.num_neurons, device=input_ids.device)
        else:
            V, S = initial_state
            
        embeddings = self.embedding(input_ids)
        w_eff_t = (self.w_rec * self.mask).t()
        logits_chunks = []
        
        if self.training:
            for t_start in range(0, seq_len, self.chunk_size):
                t_end = min(t_start + self.chunk_size, seq_len)
                chunk_emb = embeddings[:, t_start:t_end, :]
                
                chunk_logits, V, S = cp.checkpoint(
                    self._forward_chunk,
                    chunk_emb,
                    V,
                    S,
                    w_eff_t,
                    use_reentrant=False
                )
                logits_chunks.append(chunk_logits)
        else:
            for t_start in range(0, seq_len, self.chunk_size):
                t_end = min(t_start + self.chunk_size, seq_len)
                chunk_emb = embeddings[:, t_start:t_end, :]
                chunk_logits, V, S = self._forward_chunk(chunk_emb, V, S, w_eff_t)
                logits_chunks.append(chunk_logits)
                
        logits_out = torch.cat(logits_chunks, dim=1)
        return logits_out, (V.detach(), S.detach())

# 4. Mode Selection
checkpoint = None
mode = "2"

if os.path.exists(CHECKPOINT_PATH):
    print("\nExisting checkpoint detected:")
    print(" [1] Continue training from checkpoint")
    print(" [2] Start new training session (fresh initialization)")
    print(" [3] No training (Inference / Generation only)")
    choice = input("\nSelect mode (1/2/3) [default 1]: ").strip()
    if choice in ["1", "2", "3"]:
        mode = choice
    else:
        mode = "1"
else:
    print("\nNo checkpoint found. Proceeding to configure fresh training session.")
    mode = "2"

# 5. Checkpoint Metadata & Sub-Circuit Dimension
if mode in ["1", "3"]:
    print(f"\nReading metadata from '{CHECKPOINT_PATH}'...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    sub_dim = checkpoint.get("sub_dim", 12288)
    total_trained_epochs = checkpoint.get("total_epochs", 0)
    print(f"Checkpoint configured: {sub_dim:,} neurons | Previously completed: {total_trained_epochs} epochs.")
else:
    neuron_input = input("\nEnter neuron count (e.g., 8192, 12288, 16384) [default 12288]: ").strip()
    if neuron_input.isdigit() and int(neuron_input) > 0:
        sub_dim = int(neuron_input)
    else:
        sub_dim = 12288
    total_trained_epochs = 0

# 6. Extract Connectome Sub-Circuit
print(f"Loading connectome and isolating sub-circuit ({sub_dim:,} neurons)...")
scipy_csr = load_npz("fly_connectome_csr.npz")

degrees = np.array(scipy_csr.sum(axis=0)).flatten() + np.array(scipy_csr.sum(axis=1)).flatten()
top_indices = np.argsort(degrees)[-sub_dim:].copy()

sub_csr = scipy_csr[top_indices, :][:, top_indices]
dense_sub = torch.from_numpy(sub_csr.toarray()).float()

bio_mask = (dense_sub > 0).to(device)
initial_weights = (dense_sub / 25.0).clamp(0.0, 1.2).to(device)

num_synapses = int(bio_mask.sum().item())
density = (num_synapses / (sub_dim * sub_dim)) * 100
print(f"Sub-circuit mapped: {sub_dim:,} neurons, {num_synapses:,} synapses ({density:.2f}% density).")

del scipy_csr, dense_sub, sub_csr

# 7. Instantiate Model & Restore Weights
model = BioSpikingLLM(vocab_size, sub_dim, bio_mask, initial_weights, chunk_size=16).to(device)

if checkpoint is not None:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model weights successfully loaded.")

# 8. Training Execution (Modes 1 and 2)
if mode in ["1", "2"]:
    prompt_text = "\nEnter additional epochs to train [default 500]: " if mode == "1" else "\nEnter number of epochs to train [default 500]: "
    while True:
        epoch_input = input(prompt_text).strip()
        if epoch_input == "":
            epochs = 500
            break
        elif epoch_input.isdigit() and int(epoch_input) > 0:
            epochs = int(epoch_input)
            break
        else:
            print("Invalid input. Enter a positive integer.")
            
    seq_length = 96
    batch_size = 32
    learning_rate = 8e-4
    
    # Split Optimizer: SGD on recurrent matrix eliminates 1.2GB of AdamW momentum buffers
    optimizer_rec = torch.optim.SGD([model.w_rec], lr=learning_rate * 2.5)
    optimizer_dense = torch.optim.AdamW([
        {"params": model.embedding.parameters(), "lr": learning_rate},
        {"params": model.readout.parameters(), "lr": learning_rate}
    ])
    
    # Restore split optimizer states if present in checkpoint
    if mode == "1" and checkpoint is not None:
        if "optimizer_rec_state_dict" in checkpoint and "optimizer_dense_state_dict" in checkpoint:
            try:
                optimizer_rec.load_state_dict(checkpoint["optimizer_rec_state_dict"])
                optimizer_dense.load_state_dict(checkpoint["optimizer_dense_state_dict"])
                print("Restored split optimizer states from checkpoint.")
            except Exception:
                print("Notice: Optimizer parameter state mismatched; running with fresh state.")
        elif "optimizer_state_dict" in checkpoint:
            print("Notice: Migrated from single optimizer checkpoint to split SGD/AdamW configuration.")

    scheduler_rec = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_rec, T_max=epochs, eta_min=1e-5)
    scheduler_dense = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_dense, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nLaunching BPTT optimization across {sub_dim:,} neurons ({epochs} epochs, batch size {batch_size})...")
    start_time = time.time()
    model.train()
    
    current_epoch = 0
    try:
        for epoch in range(1, epochs + 1):
            current_epoch = epoch
            starts = torch.randint(0, len(encoded_data) - seq_length - 1, (batch_size,))
            inputs = torch.stack([encoded_data[i : i + seq_length] for i in starts]).to(device)
            targets = torch.stack([encoded_data[i + 1 : i + seq_length + 1] for i in starts]).to(device)
            
            optimizer_rec.zero_grad(set_to_none=True)
            optimizer_dense.zero_grad(set_to_none=True)
            
            logits, _ = model(inputs)
            
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_rec.step()
            optimizer_dense.step()
            scheduler_rec.step()
            scheduler_dense.step()
            
            with torch.no_grad():
                model.w_rec.data.masked_fill_(~model.mask, 0.0)
                
            if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
                peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
                elapsed = time.time() - start_time
                avg_per_epoch = elapsed / epoch
                current_total = total_trained_epochs + epoch
                print(f"Epoch [{epoch:4d}/{epochs}] (Total: {current_total}) | Loss: {loss.item():.4f} | Peak VRAM: {peak_vram:.1f} MB | {avg_per_epoch:.2f}s/epoch")
                
            # Autosave checkpoint every 50 epochs
            if epoch % 50 == 0:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_rec_state_dict": optimizer_rec.state_dict(),
                    "optimizer_dense_state_dict": optimizer_dense.state_dict(),
                    "sub_dim": sub_dim,
                    "vocab": chars,
                    "total_epochs": total_trained_epochs + epoch
                }, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print(f"\nTraining interrupted manually via KeyboardInterrupt at epoch {current_epoch}/{epochs}.")
        
    total_trained_epochs += current_epoch
    print(f"\nSaving latest checkpoint to '{CHECKPOINT_PATH}'...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_rec_state_dict": optimizer_rec.state_dict(),
        "optimizer_dense_state_dict": optimizer_dense.state_dict(),
        "sub_dim": sub_dim,
        "vocab": chars,
        "total_epochs": total_trained_epochs
    }, CHECKPOINT_PATH)
    print(f"Checkpoint successfully secured. Cumulative total: {total_trained_epochs} epochs.")

model.eval()

# 9. Text Generation Subroutine
def generate(model, seed="the ", max_tokens=150, temperature=0.55, top_k=4, repetition_penalty=1.4):
    model.eval()
    clean_seed = "".join([c for c in seed.lower() if c in char_to_idx])
    if len(clean_seed) == 0:
        clean_seed = "the "
        
    with torch.no_grad():
        input_ids = torch.tensor([[char_to_idx[c] for c in clean_seed]], dtype=torch.int64, device=device)
        logits, state = model(input_ids)
        
        result = clean_seed
        curr_token = input_ids[:, -1:]
        
        for _ in range(max_tokens):
            logits, state = model(curr_token, initial_state=state)
            logits = logits[:, -1, :].squeeze(0) / max(temperature, 1e-4)
            
            recent_chars = set(result[-4:])
            for rc in recent_chars:
                r_idx = char_to_idx.get(rc, None)
                if r_idx is not None:
                    if logits[r_idx] > 0:
                        logits[r_idx] /= repetition_penalty
                    else:
                        logits[r_idx] *= repetition_penalty
            
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs = np.nan_to_num(probs, nan=0.0)
            if probs.sum() == 0:
                next_idx = int(torch.argmax(logits).item())
            else:
                probs = probs / probs.sum()
                next_idx = np.random.choice(vocab_size, p=probs)
                
            next_char = idx_to_char[next_idx]
            result += next_char
            curr_token = torch.tensor([[next_idx]], dtype=torch.int64, device=device)
            
            if next_char == "." and len(result) > len(clean_seed) + 30:
                break
                
    return result

# 10. Interactive CLI
print("\n================ Model Ready ================")
print(f"Inference active on {sub_dim:,}-neuron sub-circuit.")
print("Type a prompt and press Enter (or type 'exit' to quit).\n")

while True:
    try:
        user_input = input("Enter prompt: ")
        if user_input.strip().lower() == "exit":
            break
        if len(user_input.strip()) == 0:
            continue
        output = generate(model, seed=user_input, max_tokens=150, temperature=0.55, top_k=4)
        print(f"Fly: {output}\n")
    except KeyboardInterrupt:
        break