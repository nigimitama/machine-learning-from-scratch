# ゼロから作る機械学習(1)：線形回帰



# モデル

**線形回帰**（linear regression）は，予測の目的変数$y$と特徴量（説明変数）$x_1, x_2, ..., x_d$の間に次のような線形関係を仮定したモデルを置いて予測する手法です。
$$
y = \theta_0 + \theta_1 x_1 + \cdots + \theta_d x_d + \varepsilon
$$

ここで$\theta_1, \theta_2, ..., \theta_d$は回帰係数と呼ばれるパラメータで，モデル内で推定されます。$\varepsilon$は誤差で，$\text{E}[\varepsilon]=0, \text{V}[\varepsilon]=\sigma^2$を仮定します。



サンプルサイズが$n$のデータセット$\{\boldsymbol{x}_i, y_i\}^n_{i=1}$があるとして，目的変数を$\boldsymbol{y} = (y_1, y_2, ..., y_n)^\top$，特徴量を$\boldsymbol{X}=(\boldsymbol{x}_1, \boldsymbol{x}_2, ..., \boldsymbol{x}_n)^\top$とおくと，このモデルは

$$
\boldsymbol{y}=\boldsymbol{X} \boldsymbol{\theta}+\boldsymbol{\varepsilon}
$$

と表記することができます。






# パラメータの推定

## 最小２乗法

線形回帰ではパラメータの推定に**最小二乗法**（least squares method）という方法がよく使われます。これは誤差関数$J( \boldsymbol{\theta})$を実測値$\boldsymbol{y}$と予測値$\hat{\boldsymbol{y}} = \boldsymbol{X}\hat{\boldsymbol{\theta}}$の二乗誤差の和（誤差二乗和 sum of squared error: SSE）
$$
J( \boldsymbol{\theta}) = \sum^n_{i=1} (y_i - \hat{y}_i)^2 = \sum^n_{i=1} \varepsilon_i^2 = \boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}
$$
として定義し，この二乗誤差を最小にするパラメータ（**最小二乗推定量** ordinary least square's estimator: OLSE）
$$
\newcommand{\argmin}{\mathop{\rm arg~min}\limits}

\hat{\boldsymbol{\theta}}^{LS} = \argmin_{\boldsymbol{\theta}}
\sum^n_{i=1} (y_i - \hat{y}_i)^2
$$
を採用するという方法です。

二乗誤差$(y_i - \hat{y}_i)^2 = \varepsilon_i^2$は放物線を描くため，傾きがゼロになる点が最小値だと考えられます。そのため，最小二乗法は解析的に解を求めることができます。

![img](data:image/png;base64,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)

誤差二乗和は
$$
\begin{align}
\boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}
&= (\boldsymbol{y} - \boldsymbol{X}\hat{\boldsymbol{\theta}})^\top (\boldsymbol{y} - \boldsymbol{X}\hat{\boldsymbol{\theta}})\\
&=\boldsymbol{y}^\top \boldsymbol{y}
- \boldsymbol{y}^\top\boldsymbol{X}\boldsymbol{\theta}
- (\boldsymbol{X}\boldsymbol{\theta})^\top\boldsymbol{y}
+ (\boldsymbol{X}\boldsymbol{\theta})^\top (\boldsymbol{X}\boldsymbol{\theta})\\
&= \boldsymbol{y}^\top \boldsymbol{y}
- 2 \boldsymbol{\theta}^\top \boldsymbol{X}^\top \boldsymbol{y}
+ \boldsymbol{\theta}^\top \boldsymbol{X}^\top 
\boldsymbol{X} \boldsymbol{\theta}\\
\end{align}
$$
であるから[^1]，二乗誤差の傾きがゼロになる点は

[^1]: ※転置の基本公式から，$(\boldsymbol{X}\hat{\boldsymbol{\theta}})^\top=\hat{\boldsymbol{\theta}}^\top\boldsymbol{X}^\top$，$\boldsymbol{y}^\top\boldsymbol{X}\hat{\boldsymbol{\theta}}=(\boldsymbol{X}\hat{\boldsymbol{\theta}})^\top\boldsymbol{y}=\hat{\boldsymbol{\theta}}^\top\boldsymbol{X}^\top\boldsymbol{y}$


$$
\frac{\partial \boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}}{\partial \boldsymbol{\theta}}
= -2\boldsymbol{X}^\top\boldsymbol{y}
+ 2(\boldsymbol{X}^\top\boldsymbol{X})\boldsymbol{\theta}
=\boldsymbol{0}
$$

と表すことができ，これを整理して
$$
2(\boldsymbol{X}^\top\boldsymbol{X}) \boldsymbol{\theta}
= 2\boldsymbol{X}^\top\boldsymbol{y}
$$
これの両辺を2で割ると（あるいは誤差関数の定義の際に$1/2$を掛けておくと），**正規方程式**（normal equation）とよばれる次の式が得られます。
$$
(\boldsymbol{X}^\top\boldsymbol{X})\boldsymbol{\theta}
= \boldsymbol{X}^\top\boldsymbol{y}
$$
これを$\boldsymbol{\theta}$について解けば
$$
\boldsymbol{\theta} =
(\boldsymbol{X}^\top\boldsymbol{X})^{-1}
\boldsymbol{X}^\top\boldsymbol{y}
$$
となり，最小二乗推定量$\hat{\boldsymbol{\theta}}^{LS}$が得られます。



## 実装

numpyでは，行列やベクトルの積は`@`という演算子で書くことができます。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

# サンプルデータ
X, y = load_boston(return_X_y=True)

# 最小二乗法で推定
theta = np.linalg.inv(X.T @ X) @ X.T @ y
print(theta)
```

```
[-9.28965170e-02  4.87149552e-02 -4.05997958e-03  2.85399882e+00
 -2.86843637e+00  5.92814778e+00 -7.26933458e-03 -9.68514157e-01
  1.71151128e-01 -9.39621540e-03 -3.92190926e-01  1.49056102e-02
 -4.16304471e-01]
```



scikit-learnに準拠した形[^2]でクラスとして定義するなら，以下のようになります。

```python
# scikit-learnに準拠した形で実装
from sklearn.base import BaseEstimator, RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):

    def fit(self, X, y):
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.coef_
```

[^2]: [sklearn準拠モデルの作り方 - Qiita](https://qiita.com/roronya/items/fdf35d4f69ea62e1dd91)




