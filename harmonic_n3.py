#!/usr/bin/env python3
"""
調和ガスケット (Harmonic Gasket) N=3版
2次元投影版と再帰を細かくした図の生成
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import os

# ==========================================
# Parameters
# ==========================================
N = 3
num_pts = 500_000
burn_in = 100
seed = 42
np.random.seed(seed)

print("=" * 60)
print("Harmonic Sierpinski Gasket (N=3)")
print("2D Projection & Recursive Visualization")
print("=" * 60)

# ==========================================
# Vertices in R^N
# ==========================================
qs = np.eye(N) - 1.0 / N
qs = qs * np.sqrt(2) / 2

print(f"\nVertices q_i in R^{N}:")
print(qs)
print(f"Constraint check (sum): {[qi.sum() for qi in qs]}")

# ==========================================
# Construct A_i^T matrices
# ==========================================
A1_T = np.zeros((N, N))
A1_T[0, 0] = 1.0
A1_T[0, 1:] = 2.0 / (N + 2)
for r in range(1, N):
    for c in range(1, N):
        if r == c:
            A1_T[r, c] = 2.0 / (N + 2)
        else:
            A1_T[r, c] = 1.0 / (N + 2)

print(f"\nA_1^T:")
print(A1_T)

def get_Ai_T(i):
    """Generate A_i^T by permutation"""
    if i == 0:
        return A1_T.copy()
    P = np.eye(N)
    P[[0, i]] = P[[i, 0]]
    return P @ A1_T @ P

As = np.array([get_Ai_T(i) for i in range(N)])

# ==========================================
# Chaos Game: Generate point cloud
# ==========================================
print(f"\nGenerating {num_pts:,} points via chaos game...")
x = qs[0].copy()
pts = np.empty((num_pts, N))

# Burn-in
for _ in range(burn_in):
    i = np.random.randint(N)
    x = As[i] @ (x - qs[i]) + qs[i]

# Main iteration
for k in range(num_pts):
    i = np.random.randint(N)
    x = As[i] @ (x - qs[i]) + qs[i]
    pts[k] = x

print(f"✓ Generated {num_pts:,} points")

# ==========================================
# Orthogonal projection to R^2
# Gram-Schmidt orthonormalization
# ==========================================
basis_raw = np.array([
    [1, -1, 0],
    [1, 1, -2]
], dtype=float)

Q = []
for v in basis_raw:
    w = v.copy().astype(float)
    for q in Q:
        w -= np.dot(w, q) * q
    norm_w = np.linalg.norm(w)
    if norm_w > 1e-10:
        w /= norm_w
        Q.append(w)
Q = np.array(Q)

print(f"\nOrthonormal basis (R^3 -> R^2):")
print(Q)

# Project to 2D
coords_2d = pts @ Q.T

print(f"\n2D Projected coordinates:")
print(f"  Shape: {coords_2d.shape}")
print(f"  Range: [{coords_2d.min():.4f}, {coords_2d.max():.4f}]")

# ==========================================
# FIGURE 1: 2D Projection
# ==========================================
print("\nGenerating Figure 1: 2D Projection...")

fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0a1a')
ax.set_facecolor('#0a0a1a')

ax.scatter(
    coords_2d[:, 0],
    coords_2d[:, 1],
    s=0.1,
    c='#64b5f6',
    alpha=0.4,
    rasterized=True
)

ax.spines['bottom'].set_color('#333355')
ax.spines['left'].set_color('#333355')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(colors='#666699', labelsize=9)

ax.set_title(
    'Harmonic Sierpinski Gasket (N=3, 2D Projection)',
    color='#64b5f6',
    fontsize=14,
    fontweight='bold',
    pad=15
)

ax.set_xlabel('u (first basis vector)', color='#666699', fontsize=11)
ax.set_ylabel('v (second basis vector)', color='#666699', fontsize=11)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('harmonic_gasket_2d_projection.png',
            facecolor='#0a0a1a', edgecolor='none', dpi=150, bbox_inches='tight')
print("✓ Saved: harmonic_gasket_2d_projection.png")
plt.close()

# ==========================================
# FIGURE 2: Recursive Subdivision
# ==========================================
print("\nGenerating Figure 2: Recursive Subdivision...")

def generate_recursive_triangles(depth, current_vertices=None):
    """
    Generate triangles recursively using A_i transformations

    Args:
        depth: recursion depth
        current_vertices: current triangle vertices (3x3 array)

    Returns:
        List of triangles
    """
    if current_vertices is None:
        current_vertices = qs

    triangles = []

    def recurse(d, verts):
        if d == 0:
            triangles.append(verts.copy())
            return

        # Split into 3 sub-triangles
        for i in range(3):
            qi = qs[i]
            Ai = As[i]

            next_verts = []
            for j in range(3):
                v = verts[j]
                diff = v - qi
                trans = Ai @ diff
                next_v = trans + qi
                next_verts.append(next_v)

            recurse(d - 1, np.array(next_verts))

    recurse(depth, current_vertices)
    return triangles

depths = [1, 2, 3, 4]

fig, axes = plt.subplots(2, 2, figsize=(14, 14), facecolor='#0a0a1a')
axes = axes.flatten()

for idx, depth in enumerate(depths):
    ax = axes[idx]
    ax.set_facecolor('#0a0a1a')

    # Generate triangles
    triangles_3d = generate_recursive_triangles(depth)
    triangles_2d = []

    # Project to 2D
    for tri_3d in triangles_3d:
        tri_2d = tri_3d @ Q.T
        triangles_2d.append(tri_2d)

    # Assign colors
    num_triangles = len(triangles_2d)
    colors = plt.cm.cool(np.linspace(0, 1, num_triangles))

    # Draw triangles
    for i, tri_2d in enumerate(triangles_2d):
        polygon = Polygon(
            tri_2d,
            facecolor=colors[i],
            edgecolor='#333355',
            linewidth=0.3,
            alpha=0.7
        )
        ax.add_patch(polygon)

    # Formatting
    ax.spines['bottom'].set_color('#333355')
    ax.spines['left'].set_color('#333355')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#666699', labelsize=8)

    ax.set_title(
        f'Depth = {depth}  ({num_triangles} triangles)',
        color='#64b5f6',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_aspect('equal')

    # Set limits
    all_pts = np.vstack(triangles_2d)
    margin = 0.1
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

plt.suptitle('Harmonic Sierpinski Gasket - Recursive Subdivision (2D)',
             color='#64b5f6', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

plt.savefig('harmonic_gasket_recursive.png',
            facecolor='#0a0a1a', edgecolor='none', dpi=150, bbox_inches='tight')
print("✓ Saved: harmonic_gasket_recursive.png")
plt.close()

print("\n" + "=" * 60)
print("✅ All visualizations complete!")
print("=" * 60)
print(f"\nFiles generated:")
print(f"  1. harmonic_gasket_2d_projection.png")
print(f"  2. harmonic_gasket_recursive.png")