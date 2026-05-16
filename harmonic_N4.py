#!/usr/bin/env python3
"""
調和ガスケット (Harmonic Gasket) の3D描画
縮小写像（IFS）を使った生成
ローカル環境対応版
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import os

class HarmonicGasket:
    """調和ガスケット生成クラス"""

    def __init__(self):
        # 4次元の頂点（四面体の頂点）
        self.qs = np.array([
            [0.75, -0.25, -0.25, -0.25],
            [-0.25, 0.75, -0.25, -0.25],
            [-0.25, -0.25, 0.75, -0.25],
            [-0.25, -0.25, -0.25, 0.75]
        ])

        # 初期変換行列
        self.A0 = np.array([
            [1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0],
            [0.0, 1.0/3.0, 1.0/6.0, 1.0/6.0],
            [0.0, 1.0/6.0, 1.0/3.0, 1.0/6.0],
            [0.0, 1.0/6.0, 1.0/6.0, 1.0/3.0]
        ])

        # 各変換行列を生成
        self.As = [self._get_ai(i) for i in range(4)]

        # 射影行列 Q (4D -> 3D)
        self.Q = np.array([
            [1/np.sqrt(2), -1/np.sqrt(2), 0, 0],
            [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6), 0],
            [1/np.sqrt(12), 1/np.sqrt(12), 1/np.sqrt(12), -3/np.sqrt(12)]
        ])

    def _get_ai(self, i):
        """i番目の変換行列を生成"""
        A = self.A0.copy()
        # 行と列をスワップ
        if i != 0:
            A[[0, i]] = A[[i, 0]]  # 行のスワップ
            A[:, [0, i]] = A[:, [i, 0]]  # 列のスワップ
        return A

    def project(self, v4):
        """4次元の点を3次元に射影"""
        v3 = self.Q @ v4
        return v3 * 3

    def generate_ifs(self, depth, current_vertices=None):
        """
        縮小写像で調和ガスケットを生成

        Args:
            depth: 再帰の深さ
            current_vertices: 現在の頂点（4×4配列）

        Returns:
            三角面のリスト: [(3つの3D点), ...]
        """
        if current_vertices is None:
            current_vertices = self.qs

        faces = []
        self._generate_recursive(depth, current_vertices, faces)
        return faces

    def _generate_recursive(self, depth, current_vertices, faces):
        """再帰的に調和ガスケットを生成"""
        if depth == 0:
            # 四面体の4つの面を生成
            face_indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
            for indices in face_indices:
                # 4D点を3Dに射影
                projected = [self.project(current_vertices[i]) for i in indices]
                faces.append(projected)
            return

        # 4つの部分に再帰
        for i in range(4):
            next_vertices = []
            qi = self.qs[i]
            Ai = self.As[i]

            for j in range(4):
                v = current_vertices[j]
                diff = v - qi
                trans = Ai @ diff
                next_v = trans + qi
                next_vertices.append(next_v)

            self._generate_recursive(depth - 1, np.array(next_vertices), faces)

    def get_face_colors(self, num_faces):
        """
        面の数に応じた色を生成
        同じ深さの再帰レベルは同じ色系で分類
        """
        colors = []
        # HSV色空間で色を生成
        for i in range(num_faces):
            # 面のインデックスに基づいて色を決定
            hue = (i % 4) / 4.0  # 0-1の範囲でHueを循環
            saturation = 0.8
            value = 0.7

            # HSVをRGBに変換
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)

        return colors

    def render_3d(self, depth=2, output_path='harmonic_gasket.png', figsize=(12, 10), dpi=150):
        """
        3DでレンダリングしてPNGに保存

        Args:
            depth: 再帰の深さ
            output_path: 出力ファイルパス
            figsize: 図のサイズ
            dpi: 解像度
        """
        print(f"深さ {depth} で調和ガスケットを生成中...")

        # ガスケット生成
        faces = self.generate_ifs(depth)
        print(f"生成された面の数: {len(faces)}")

        # 色生成
        colors = self.get_face_colors(len(faces))

        # 3Dプロット設定
        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

        # 面を描画
        poly_vertices = np.array(faces)
        poly_collection = Poly3DCollection(
            poly_vertices,
            facecolors=colors,
            edgecolors='#222222',
            linewidths=0.2,
            alpha=0.95
        )
        ax.add_collection3d(poly_collection)

        # 軸の設定
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])

        # 見た目調整
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('#333333')
        ax.yaxis.pane.set_edgecolor('#333333')
        ax.zaxis.pane.set_edgecolor('#333333')

        ax.tick_params(colors='#666666', labelsize=8)
        ax.set_xlabel('X', color='#666666', fontsize=10)
        ax.set_ylabel('Y', color='#666666', fontsize=10)
        ax.set_zlabel('Z', color='#666666', fontsize=10)

        # タイトル
        ax.set_title(
            f'Harmonic Gasket (N=4, Depth={depth})',
            color='#64b5f6',
            fontsize=16,
            fontweight='bold',
            pad=20
        )

        # カメラアングル
        ax.view_init(elev=20, azim=45)

        # レイアウト調整
        plt.tight_layout()

        # 出力ディレクトリ作成
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存
        print(f"PNG保存中: {output_path}")
        plt.savefig(
            output_path,
            facecolor='#0a0a0a',
            edgecolor='none',
            dpi=dpi,
            bbox_inches='tight'
        )
        print(f"✓ 完了: {output_path}")

        plt.close()

    def render_multiple_angles(self, depth=2, output_dir='./output'):
        """
        複数の角度からレンダリング
        """
        os.makedirs(output_dir, exist_ok=True)

        angles = [
            (20, 45, "front"),
            (30, 120, "right"),
            (10, 225, "back"),
            (60, 45, "top")
        ]

        faces = self.generate_ifs(depth)
        colors = self.get_face_colors(len(faces))

        for elev, azim, label in angles:
            fig = plt.figure(figsize=(10, 10), facecolor='#0a0a0a')
            ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')

            poly_vertices = np.array(faces)
            poly_collection = Poly3DCollection(
                poly_vertices,
                facecolors=colors,
                edgecolors='#222222',
                linewidths=0.2,
                alpha=0.95
            )
            ax.add_collection3d(poly_collection)

            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.set_zlim([-3, 3])

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            ax.xaxis.pane.set_edgecolor('#333333')
            ax.yaxis.pane.set_edgecolor('#333333')
            ax.zaxis.pane.set_edgecolor('#333333')

            ax.tick_params(colors='#666666', labelsize=8)

            ax.set_title(
                f'Harmonic Gasket - {label.capitalize()}',
                color='#64b5f6',
                fontsize=14,
                fontweight='bold'
            )

            ax.view_init(elev=elev, azim=azim)

            plt.tight_layout()
            output_path = f'{output_dir}/harmonic_gasket_{label}_depth{depth}.png'
            plt.savefig(
                output_path,
                facecolor='#0a0a0a',
                edgecolor='none',
                dpi=150,
                bbox_inches='tight'
            )
            print(f"✓ 保存: {output_path}")
            plt.close()


def main():
    """メイン処理"""
    gasket = HarmonicGasket()

    # 出力ディレクトリ
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # 深さ2のガスケット
    gasket.render_3d(depth=2, output_path=f'{output_dir}/harmonic_gasket_depth2.png')

    # 深さ3のガスケット
    gasket.render_3d(depth=3, output_path=f'{output_dir}/harmonic_gasket_depth3.png')

    # 深さ4のガスケット（時間がかかる）
    print("\n深さ4の生成は時間がかかる可能性があります...")
    gasket.render_3d(depth=4, output_path=f'{output_dir}/harmonic_gasket_depth4.png', dpi=150)

    # 複数角度レンダリング
    print("\n複数の角度からのレンダリング...")
    gasket.render_multiple_angles(depth=3, output_dir=output_dir)

    print("\n✅ すべての生成が完了しました！")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == '__main__':
    main()