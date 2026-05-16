import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# ==========================================
# Parameters
# ==========================================
N = 4                  # N=4 の harmonic gasket
num_pts = 500_000      # 点数 (※重い場合は減らしてください)
burn_in = 100
seed = 0
np.random.seed(seed)

# ==========================================
# Vertices q_i in R^N
# q_i = e_i - (1/N) 1
# ==========================================
qs = np.eye(N) - 1.0 / N

# ==========================================
# Construct A_1^T
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

# ==========================================
# Generate A_i^T by permutation
# ==========================================
def get_Ai_T(i):
    if i == 0:
        return A1_T.copy()
    P = np.eye(N)
    P[[0, i]] = P[[i, 0]]
    return P @ A1_T @ P

As = np.array([get_Ai_T(i) for i in range(N)])

# ==========================================
# Chaos game
# ==========================================
x = qs[0].copy()
pts = np.empty((num_pts, N))

for _ in range(burn_in):
    i = np.random.randint(N)
    x = As[i] @ (x - qs[i]) + qs[i]

for k in range(num_pts):
    i = np.random.randint(N)
    x = As[i] @ (x - qs[i]) + qs[i]
    pts[k] = x

# ==========================================
# R^4 の超平面 {sum x_i = 0} の正規直交基底
# ==========================================
basis = np.array([
    [1, -1,  0,  0],
    [1,  1, -2,  0],
    [1,  1,  1, -3]
], dtype=float)

Q = []
for v in basis:
    w = v.copy()
    for q in Q:
        w -= np.dot(w, q) * q
    w /= np.linalg.norm(w)
    Q.append(w)
Q = np.array(Q)

# ==========================================
# 4次元 -> 3次元
# ==========================================
coords = pts @ Q.T

# ==========================================
# Plot with Sliders
# ==========================================
fig = plt.figure(figsize=(10, 10))
# スライダー用の余白を確保
plt.subplots_adjust(bottom=0.25)
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    coords[:, 0],
    coords[:, 1],
    coords[:, 2],
    s=0.01,
    c="black",
    marker=".",
    alpha=0.5,
    rasterized=True
)

# 初期視点の設定
init_elev = 20
init_azim = 35
ax.view_init(elev=init_elev, azim=init_azim)
ax.set_axis_off()

# 各軸のスケールを揃える
mins = coords.min(axis=0)
maxs = coords.max(axis=0)
center = (mins + maxs) / 2
radius = (maxs - mins).max() / 2
ax.set_xlim(center[0] - radius, center[0] + radius)
ax.set_ylim(center[1] - radius, center[1] + radius)
ax.set_zlim(center[2] - radius, center[2] + radius)

# ==========================================
# Slider Setup
# ==========================================
# スライダーの配置 (left, bottom, width, height)
axcolor = 'lightgoldenrodyellow'
ax_elev = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_azim = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor=axcolor)

# スライダーオブジェクトの作成
s_elev = Slider(ax_elev, 'Elevation', -90.0, 90.0, valinit=init_elev)
s_azim = Slider(ax_azim, 'Azimuth', 0.0, 360.0, valinit=init_azim)

# スライダーが動いた時の更新関数
def update(val):
    ax.view_init(elev=s_elev.val, azim=s_azim.val)
    fig.canvas.draw_idle()

# イベントの紐付け
s_elev.on_changed(update)
s_azim.on_changed(update)

print("GUIウィンドウを起動します。ウィンドウ下部のスライダーで回転できます。")
plt.show()