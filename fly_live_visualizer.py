# fly_live_visualizer.py
import sys
import os
import time
import threading
import queue
import gzip
import csv
import re
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import load_npz

from vispy import app, scene
from vispy.scene import visuals

# 1. Hardware & Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "fly_brain_checkpoint.pt"
COORDS_PATH = "coordinates.csv.gz"

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"'{CHECKPOINT_PATH}' not found. Train the model first.")
if not os.path.exists(COORDS_PATH):
    raise FileNotFoundError(f"'{COORDS_PATH}' not found in the working directory.")

# 2. FastSigmoid Surrogate & Architecture
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

class BioSpikingLLM(nn.Module):
    def __init__(self, vocab_size, num_neurons, bio_mask, initial_weights, chunk_size=16):
        super().__init__()
        self.num_neurons = num_neurons
        self.chunk_size = chunk_size
        self.register_buffer("mask", bio_mask)
        
        self.w_rec = nn.Parameter(initial_weights.clone())
        self.embedding = nn.Embedding(vocab_size, num_neurons)
        self.readout = nn.Sequential(
            nn.Dropout(0.10),
            nn.Linear(num_neurons, 512),
            nn.GELU(),
            nn.Linear(512, vocab_size)
        )
        self.leak = 0.88
        self.v_thresh = 1.0

    def forward_step(self, input_id, V, S):
        w_eff_t = (self.w_rec * self.mask).t()
        I_ext = self.embedding(input_id).view(-1, self.num_neurons)
        syn_input = torch.matmul(S, w_eff_t)
        
        V = (V * self.leak * (1.0 - S)) + syn_input + I_ext
        S = surrogate_spike(V - self.v_thresh)
        logits = self.readout(S)
        return logits, V, S

# 3. Load Checkpoint & Connectome
print(f"Loading checkpoint from '{CHECKPOINT_PATH}'...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
sub_dim = checkpoint.get("sub_dim", 12288)
chars = checkpoint.get("vocab", list("abcdefghijklmnopqrstuvwxyz .,!?-"))
vocab_size = len(chars)
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

print(f"Loading connectome sub-circuit ({sub_dim:,} neurons)...")
scipy_csr = load_npz("fly_connectome_csr.npz")
degrees = np.array(scipy_csr.sum(axis=0)).flatten() + np.array(scipy_csr.sum(axis=1)).flatten()
top_indices = np.argsort(degrees)[-sub_dim:].copy()

sub_csr = scipy_csr[top_indices, :][:, top_indices]
dense_sub = torch.from_numpy(sub_csr.toarray()).float()
bio_mask = (dense_sub > 0).to(device)
initial_weights = (dense_sub / 25.0).clamp(0.0, 1.2).to(device)

model = BioSpikingLLM(vocab_size, sub_dim, bio_mask, initial_weights).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
del scipy_csr, dense_sub, sub_csr

# 4. Universal Coordinate Extraction
print(f"Reading biological data from '{COORDS_PATH}'...")

raw_coords = []
region_tags = []

with gzip.open(COORDS_PATH, mode="rt", encoding="utf-8", errors="replace") as f:
    sample = [f.readline() for _ in range(25)]
    f.seek(0)
    
    # Skip any comment blocks
    data_lines = [l.strip() for l in sample if l.strip() and not l.strip().startswith('#')]
    if not data_lines:
        raise ValueError(f"'{COORDS_PATH}' contains no data rows.")
        
    first_data_line = data_lines[0]
    delim = '\t' if '\t' in first_data_line else (';' if ';' in first_data_line else ',')
    reader = csv.reader(f, delimiter=delim)
    
    first_row = next(reader)
    while first_row and (len(first_row) == 0 or first_row[0].strip().startswith('#')):
        first_row = next(reader)
        
    header = [c.strip().lower() for c in first_row]
    print(f"Detected columns: {first_row}")
    
    idx_x, idx_y, idx_z = None, None, None
    vec_col = None
    neuropil_col = None
    
    for i, c in enumerate(header):
        if c in ['x', 'pt_x', 'x_pt', 'pos_x']:
            idx_x = i
        elif c in ['y', 'pt_y', 'y_pt', 'pos_y']:
            idx_y = i
        elif c in ['z', 'pt_z', 'z_pt', 'pos_z']:
            idx_z = i
        elif any(k in c for k in ['position', 'coordinate', 'location', 'centroid', 'pt_position']):
            vec_col = i
        elif any(k in c for k in ['neuropil', 'region', 'area', 'class', 'cell_type']):
            neuropil_col = i

    # Fallback substring scan if explicit matches not found
    if idx_x is None or idx_y is None or idx_z is None:
        cand_x = [i for i, c in enumerate(header) if 'x' in c]
        cand_y = [i for i, c in enumerate(header) if 'y' in c]
        cand_z = [i for i, c in enumerate(header) if 'z' in c]
        if cand_x and cand_y and cand_z:
            idx_x, idx_y, idx_z = cand_x[0], cand_y[0], cand_z[0]

    # Process first row if it was already data (headerless file)
    num_vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", " ".join(first_row))
    if len(num_vals) >= 3 and idx_x is None and vec_col is None:
        if len(num_vals) >= 4:
            raw_coords.append([float(num_vals[1]), float(num_vals[2]), float(num_vals[3])])
        else:
            raw_coords.append([float(num_vals[0]), float(num_vals[1]), float(num_vals[2])])
        region_tags.append("")

    for row in reader:
        if not row:
            continue
        try:
            if idx_x is not None and idx_y is not None and idx_z is not None:
                if len(row) > max(idx_x, idx_y, idx_z):
                    raw_coords.append([float(row[idx_x]), float(row[idx_y]), float(row[idx_z])])
                    tag = row[neuropil_col].strip().upper() if (neuropil_col and len(row) > neuropil_col) else ""
                    region_tags.append(tag)
            elif vec_col is not None and len(row) > vec_col:
                parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row[vec_col])
                if len(parts) >= 3:
                    raw_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    tag = row[neuropil_col].strip().upper() if (neuropil_col and len(row) > neuropil_col) else ""
                    region_tags.append(tag)
            else:
                parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", " ".join(row))
                if len(parts) >= 4:
                    raw_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    region_tags.append("")
                elif len(parts) == 3:
                    raw_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    region_tags.append("")
        except Exception:
            continue

raw_xyz = np.array(raw_coords, dtype=np.float32)
if len(raw_xyz) == 0:
    raise ValueError(f"Could not extract coordinates from '{COORDS_PATH}'. Check file structure.")

# Voxel anisotropy correction (FlyWire 40nm Z vs 4nm XY)
x_span = raw_xyz[:, 0].max() - raw_xyz[:, 0].min()
z_span = raw_xyz[:, 2].max() - raw_xyz[:, 2].min()
if (z_span / max(x_span, 1e-4)) < 0.15:
    raw_xyz[:, 2] *= 10.0

center = np.median(raw_xyz, axis=0)
centered = raw_xyz - center
scale = np.max(np.ptp(centered, axis=0))
norm_xyz = (centered / scale) * 340.0
norm_xyz[:, 1] = -norm_xyz[:, 1]

# 5. Neuropil Spatial Classification Palette
def assign_neuropil_colors(xyz, tags):
    n_pts = len(xyz)
    colors = np.zeros((n_pts, 3), dtype=np.float32)
    
    PALETTE = {
        "OPTIC_LEFT": np.array([0.00, 0.85, 0.90]),    # Electric Cyan (Left Eye / Medulla)
        "OPTIC_RIGHT": np.array([0.72, 0.28, 0.98]),   # Vivid Violet (Right Eye / Medulla)
        "CENTRAL_CX": np.array([0.20, 0.98, 0.35]),    # Neon Green (Central Complex)
        "MUSHROOM_BODY": np.array([1.00, 0.58, 0.05]), # Vibrant Amber (Memory / KC)
        "ANTENNAL_LOBE": np.array([0.98, 0.18, 0.28]), # Crimson / Coral (Olfaction)
        "SEZ_GNATHAL": np.array([0.05, 0.80, 0.65]),   # Teal (Gnathal Ganglion)
        "SUPERIOR_PROT": np.array([0.95, 0.25, 0.80]), # Hot Magenta (Superior Protocerebrum)
        "LATERAL_HORN": np.array([0.98, 0.88, 0.15])   # Gold (Sensory / Lateral Horn)
    }

    has_valid_tags = any(len(t) > 0 for t in tags[:1000])

    for i in range(n_pts):
        tag = tags[i] if has_valid_tags else ""
        x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        
        if "ME_L" in tag or "LO_L" in tag or "LOP_L" in tag:
            colors[i] = PALETTE["OPTIC_LEFT"]
        elif "ME_R" in tag or "LO_R" in tag or "LOP_R" in tag:
            colors[i] = PALETTE["OPTIC_RIGHT"]
        elif any(k in tag for k in ["FB", "EB", "PB", "NO", "CX"]):
            colors[i] = PALETTE["CENTRAL_CX"]
        elif any(k in tag for k in ["MB", "KC", "CA", "PED"]):
            colors[i] = PALETTE["MUSHROOM_BODY"]
        elif "AL" in tag:
            colors[i] = PALETTE["ANTENNAL_LOBE"]
        elif any(k in tag for k in ["SEZ", "GNG", "AMMC"]):
            colors[i] = PALETTE["SEZ_GNATHAL"]
        elif any(k in tag for k in ["SMP", "SLP", "SIP"]):
            colors[i] = PALETTE["SUPERIOR_PROT"]
        elif any(k in tag for k in ["LH", "AVLP", "PVLP"]):
            colors[i] = PALETTE["LATERAL_HORN"]
        else:
            if x < -62.0:
                colors[i] = PALETTE["OPTIC_LEFT"]
            elif x > 62.0:
                colors[i] = PALETTE["OPTIC_RIGHT"]
            elif abs(x) <= 28.0 and abs(y) <= 28.0 and z >= -12.0:
                colors[i] = PALETTE["CENTRAL_CX"]
            elif abs(x) <= 32.0 and y < -28.0 and z > 5.0:
                colors[i] = PALETTE["ANTENNAL_LOBE"]
            elif y < -35.0:
                colors[i] = PALETTE["SEZ_GNATHAL"]
            elif 26.0 < abs(x) <= 58.0 and y > 15.0 and z < 10.0:
                colors[i] = PALETTE["MUSHROOM_BODY"]
            elif y > 35.0:
                colors[i] = PALETTE["SUPERIOR_PROT"]
            else:
                colors[i] = PALETTE["LATERAL_HORN"]
                
    return colors

# Map active neurons and background anatomical volume
if len(norm_xyz) >= sub_dim:
    active_coords = norm_xyz[:sub_dim].copy()
    active_tags = region_tags[:sub_dim]
else:
    reps = int(np.ceil(sub_dim / len(norm_xyz)))
    active_coords = np.tile(norm_xyz, (reps, 1))[:sub_dim].copy()
    active_tags = (region_tags * reps)[:sub_dim]

bg_stride = max(1, len(norm_xyz) // 28000)
ghost_coords = norm_xyz[::bg_stride].copy()
ghost_tags = region_tags[::bg_stride]
num_ghost = len(ghost_coords)

active_base_rgb = assign_neuropil_colors(active_coords, active_tags)
ghost_base_rgb = assign_neuropil_colors(ghost_coords, ghost_tags)

print(f"Extracted {len(raw_xyz):,} biological coordinates.")
print(f"Segmented {sub_dim:,} active neurons across 8 anatomical neuropils.")

# 6. VisPy Scene Initialization
canvas = scene.SceneCanvas(
    keys='interactive',
    size=(1360, 850),
    show=True,
    bgcolor='#05070c',
    title='Drosophila Melanogaster Connectome - Anatomical Neuropil Spike Visualizer'
)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, elevation=15, distance=420)

# Ghost Anatomical Volume
ghost_markers = visuals.Markers()
ghost_markers.set_gl_state(blend=True, depth_test=False)
view.add(ghost_markers)

ghost_rgba = np.zeros((num_ghost, 4), dtype=np.float32)
ghost_rgba[:, :3] = ghost_base_rgb * 0.75
ghost_rgba[:, 3] = 0.08
ghost_sizes = np.full(num_ghost, 2.0, dtype=np.float32)
ghost_markers.set_data(ghost_coords, edge_color=None, face_color=ghost_rgba, size=ghost_sizes)

# Active SNN Neuron Field
active_markers = visuals.Markers()
active_markers.set_gl_state(blend=True, depth_test=False)
view.add(active_markers)

active_rgba = np.zeros((sub_dim, 4), dtype=np.float32)
active_rgba[:, :3] = active_base_rgb
active_rgba[:, 3] = 0.28
active_sizes = np.full(sub_dim, 2.5, dtype=np.float32)
active_markers.set_data(active_coords, edge_color=None, face_color=active_rgba, size=active_sizes)

# 7. Generation Worker
spike_queue = queue.Queue()
spike_activity = np.zeros(sub_dim, dtype=np.float32)

def generation_worker():
    time.sleep(1.0)
    print("\n" + "="*60)
    print("FLY CONNECTOME ANATOMICAL PROMPT READY")
    print("Type your message and press Enter (or 'exit' to quit).")
    print("="*60 + "\n")
    
    while True:
        try:
            sys.stdout.write("Prompt: ")
            sys.stdout.flush()
            user_input = sys.stdin.readline()
            if not user_input or user_input.strip().lower() == "exit":
                break
            
            clean_input = "".join([c for c in user_input.lower() if c in char_to_idx]).strip()
            if not clean_input:
                continue
                
            prompt = clean_input + " "
            V = torch.zeros(1, sub_dim, device=device)
            S = torch.zeros(1, sub_dim, device=device)
            
            with torch.no_grad():
                for ch in prompt:
                    token_id = torch.tensor([[char_to_idx[ch]]], dtype=torch.int64, device=device)
                    logits, V, S = model.forward_step(token_id, V, S)
                    spike_queue.put(S.squeeze().detach().cpu().numpy())
                    time.sleep(0.015)
                
                sys.stdout.write("Fly: " + prompt)
                sys.stdout.flush()
                
                curr_token = torch.tensor([[char_to_idx[prompt[-1]]]], dtype=torch.int64, device=device)
                generated_chars = 0
                response = ""
                
                for _ in range(150):
                    logits, V, S = model.forward_step(curr_token, V, S)
                    
                    spike_data = S.squeeze().detach().cpu().numpy()
                    spike_queue.put(spike_data)
                    
                    logits = logits.squeeze() / 0.50
                    for rc in set(response[-4:]):
                        r_idx = char_to_idx.get(rc, None)
                        if r_idx is not None:
                            logits[r_idx] = logits[r_idx] / 1.4 if logits[r_idx] > 0 else logits[r_idx] * 1.4
                    
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    probs = np.nan_to_num(probs, nan=0.0)
                    next_idx = int(np.random.choice(vocab_size, p=probs / probs.sum())) if probs.sum() > 0 else int(torch.argmax(logits).item())
                    next_char = idx_to_char[next_idx]
                    
                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                    response += next_char
                    generated_chars += 1
                    curr_token = torch.tensor([[next_idx]], dtype=torch.int64, device=device)
                    
                    time.sleep(0.04)
                    
                    if next_char == "." and generated_chars > 25:
                        break
                        
                sys.stdout.write("\n\n")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"\nWorker exception: {e}")
            break

# 8. Real-Time Render Loop
def update(ev):
    global spike_activity, active_rgba, active_sizes
    
    while not spike_queue.empty():
        spikes = spike_queue.get_nowait().flatten()
        spike_activity = np.maximum(spike_activity, spikes)
        
    view.camera.azimuth += 0.12
    spike_activity *= 0.85
    
    active_rgba[:, :3] = active_base_rgb
    active_rgba[:, 3] = 0.28
    active_sizes[:] = 2.5
    
    active_idx = spike_activity > 0.02
    if np.any(active_idx):
        act = spike_activity[active_idx, np.newaxis]
        flash_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        active_rgba[active_idx, :3] = (active_base_rgb[active_idx] * (1.0 - act)) + (flash_rgb * act)
        active_rgba[active_idx, 3] = np.clip(0.3 + (act[:, 0] * 0.7), 0.3, 1.0)
        active_sizes[active_idx] = 2.5 + (act[:, 0] * 7.5)

    active_markers.set_data(active_coords, edge_color=None, face_color=active_rgba, size=active_sizes)

timer = app.Timer(interval=1/60.0, connect=update, start=True)
worker_thread = threading.Thread(target=generation_worker, daemon=True)
worker_thread.start()

if __name__ == '__main__':
    app.run()