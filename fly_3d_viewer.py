# fly_3d_viewer.py
import numpy as np
import torch
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh
from vispy import app, scene
from vispy.color import ColorArray

# 1. Load Sub-Circuit Topology
print("Loading connectome topology...")
scipy_csr = load_npz("fly_connectome_csr.npz")
sub_dim = 8192

degrees = np.array(scipy_csr.sum(axis=0)).flatten() + np.array(scipy_csr.sum(axis=1)).flatten()
top_indices = np.argsort(degrees)[-sub_dim:].copy()
sub_csr = scipy_csr[top_indices, :][:, top_indices]

# 2. Derive 3D Coordinates via Normalized Laplacian Eigenmaps
print("Calculating 3D morphological coordinates via Spectral Embedding...")
adj = (sub_csr + sub_csr.T).astype(float)
deg = np.array(adj.sum(axis=1)).flatten()
deg[deg == 0] = 1.0
inv_sqrt_deg = 1.0 / np.sqrt(deg)

from scipy.sparse import diags
D_inv = diags(inv_sqrt_deg)
L_norm = diags(np.ones(sub_dim)) - D_inv.dot(adj).dot(D_inv)

# Extract lowest non-trivial eigenvectors as (x, y, z) spatial coordinates
eigenvalues, eigenvectors = eigsh(L_norm, k=4, which='SM')
coords_3d = eigenvectors[:, 1:4]
coords_3d = (coords_3d - coords_3d.mean(axis=0)) / (coords_3d.std(axis=0) + 1e-6)
coords_3d *= 250.0  # Scale into viewport space

# 3. Setup VisPy OpenGL Scene
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='#050811', size=(1280, 800))
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.distance = 700

scatter = scene.visuals.Markers()
view.add(scatter)

# 4. Spiking Activity State Buffer
activity = np.zeros(sub_dim, dtype=np.float32)
colors = np.zeros((sub_dim, 4), dtype=np.float32)
colors[:, 0] = 0.15  # R
colors[:, 1] = 0.35  # G
colors[:, 2] = 0.65  # B
colors[:, 3] = 0.25  # Alpha

# 5. Live Simulation Loop (60 FPS Update)
decay = 0.86

def update(event):
    global activity, colors
    
    # Generate sparse burst events (or feed S from model.forward)
    spikes = (np.random.rand(sub_dim) < 0.02).astype(np.float32)
    activity = (activity * decay) + spikes
    activity = np.clip(activity, 0.0, 1.0)
    
    # Dynamic Color Transition: Dim blue-grey -> Glowing amber/white
    colors[:, 0] = 0.15 + (0.85 * activity)
    colors[:, 1] = 0.35 + (0.45 * activity)
    colors[:, 2] = 0.65 - (0.40 * activity)
    colors[:, 3] = np.clip(0.20 + (0.80 * activity), 0.1, 1.0)
    
    sizes = 4.0 + (10.0 * activity)
    
    scatter.set_data(coords_3d, edge_width=0, face_color=colors, size=sizes)
    view.camera.azimuth += 0.25  # Smooth continuous auto-rotation

timer = app.Timer(interval=1/60.0, connect=update, start=True)

if __name__ == '__main__':
    print("Launching real-time 3D biological connectome visualizer...")
    app.run()