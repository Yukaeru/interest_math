import numpy as np
import matplotlib.pyplot as plt

def semicircle_fractal(x, n):
    if n == 0:
        return np.zeros_like(x)

    # 区間 [-2, 2] を 2^n 個に分割
    num_intervals = 2**n
    r = 2 / num_intervals
    y = np.zeros_like(x)

    for k in range(num_intervals):
        # 各小区間の範囲
        a = -2 + (4 * k) / num_intervals
        b = -2 + (4 * (k + 1)) / num_intervals
        c = (a + b) / 2

        # マスクを作成
        mask = (x >= a) & (x <= b)

        # 半円の計算 (虚数回避のためにclip)
        inner = r**2 - (x[mask] - c)**2
        inner = np.maximum(inner, 0)

        # 交互に上下を入れ替え
        sign = (-1)**k
        y[mask] = sign * np.sqrt(inner)
    return y

# 描画設定
x = np.linspace(-2, 2, 1000)
plt.figure(figsize=(10, 6))

for n in [1, 2, 3, 4,6,8]:
    plt.plot(x, semicircle_fractal(x, n), label=f'n={n}')

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.title("Recursive Semicircles Convergence ($L(f_n) = 2\pi$ vs $L(f)=4$)")
plt.xlabel("x")
plt.ylabel("f_n(x)")
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()