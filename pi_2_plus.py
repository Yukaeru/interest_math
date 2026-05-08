import numpy as np
import matplotlib.pyplot as plt

def semicircle_fractal_and_derivative(x, n):
    if n == 0:
        return np.zeros_like(x), np.zeros_like(x)

    # 区間 [-2, 2] を 2^n 個に分割
    num_intervals = 2**n
    r = 2 / num_intervals
    y = np.zeros_like(x)
    y_prime = np.zeros_like(x)

    for k in range(num_intervals):
        # 各小区間の範囲
        a = -2 + (4 * k) / num_intervals
        b = -2 + (4 * (k + 1)) / num_intervals
        c = (a + b) / 2

        # マスクを作成 (端点付近を除外して発散を制御)
        # グリッドの細かさに合わせて微調整
        epsilon = 1e-6
        mask = (x >= a + epsilon) & (x <= b - epsilon)

        # --- 元の関数 f_n(x) ---
        inner = r**2 - (x[mask] - c)**2
        inner = np.maximum(inner, 0) # 虚数回避
        sign = (-1)**k
        y[mask] = sign * np.sqrt(inner)

        # --- 微分関数 f_n'(x) ---
        # 端点での発散を制御するために、分母に小さな値を足す
        sqrt_inner = np.sqrt(inner + 1e-9)
        deriv = - (x[mask] - c) / sqrt_inner

        y_prime[mask] = sign * deriv

    return y, y_prime

# 描画設定
num_points = 5000 # 微細な構造を捉えるために細かく
x = np.linspace(-2, 2, num_points)
ns = [1, 3, 6]

# サブプロットの作成 (2xlen(ns))
fig, axes = plt.subplots(2, len(ns), figsize=(15, 8), sharex=True)

for i, n in enumerate(ns):
    y, y_prime = semicircle_fractal_and_derivative(x, n)

    # --- 元の関数 f_n(x) のプロット ---
    ax_f = axes[0, i]
    ax_f.plot(x, y, color='blue')
    ax_f.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_f.set_title(f'$n={n}$\n$f_n(x)$')
    ax_f.axis('equal')
    ax_f.grid(True, alpha=0.3)

    # --- 微分関数 f_n'(x) のプロット ---
    ax_df = axes[1, i]
    ax_df.plot(x, y_prime, color='orange')
    ax_df.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax_df.set_title(f'$n={n}$\n$f_n\'(x)$')
    ax_df.grid(True, alpha=0.3)

    # y軸の範囲を制限して、発散する挙動を視覚化
    ax_df.set_ylim([-50, 50]) # 適当な範囲で制限

    # 不連続点（各小区間の端点）に縦線を引く (視覚的な不連続性を強調)
    for k in range(1, 2**n):
        split_point = -2 + (4 * k) / (2**n)
        ax_df.axvline(split_point, color='gray', linewidth=0.3, linestyle='-')

# 全体設定
fig.suptitle("Recursive Semicircles and Their Derivatives", fontsize=16)
for i in range(len(ns)):
    axes[1, i].set_xlabel("x")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()