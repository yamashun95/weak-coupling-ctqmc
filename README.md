# weak-coupling-ctqmc

弱結合連続時間量子モンテカルロ法（CTQMC）による格子模型のシミュレータです。  
本実装は、以下の教科書の第8.4章に基づいて構成されています：

> J.E. Gubernatis, N. Kawashima, and P. Werner,  
> *Quantum Monte Carlo Methods: Algorithms for Lattice Models*,  
> Cambridge University Press, 2016.

特に、相互作用展開（interaction expansion）と補助場導入（Assaad–Lang法）を用いたCTQMCアルゴリズム、および第8.4.3節に記載されたグリーン関数 \( G(\tau) \) の測定手法を含んでいます。

---

## 🔧 特徴

- 弱結合CTQMC（interaction expansion）
- 補助Ising場の導入（Assaad–Lang補助場変換）
- 頂点の挿入／削除に基づくMetropolisサンプリング
- 高速な行列更新（rank-1更新）
- 時間グリーン関数 \( G(\tau) \) の測定
- Weiss Green関数（例えばBethe格子）の入力に対応

---

## 📦 インストール

```bash
pip install .
```

または開発モードでインストール：

```bash
pip install -e .
```

---

## 🚀 使い方

基本的なシミュレーションは、以下のように `run_simulation.py` を実行することで開始できます：

```bash
python run_simulation.py
```

出力として、グリーン関数 \( G_\sigma(\tau) \) が測定されます。  
必要に応じてフーリエ変換して \( G(i\omega_n) \) に変換可能です。

---

## 🧪 依存関係

- Python >= 3.10
- NumPy
- SciPy
- Matplotlib（プロット用・任意）

---

## 📚 参考文献

- Gubernatis, J.E., Kawashima, N., & Werner, P. (2016).  
  *Quantum Monte Carlo Methods: Algorithms for Lattice Models*, Cambridge University Press.
- Assaad, F. F., & Lang, T. C. (2007).  
  *Diagrammatic determinantal quantum Monte Carlo methods...*, Phys. Rev. B 76, 035116.

---
