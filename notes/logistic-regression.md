# ゼロから作る機械学習(2)：ロジスティック回帰





# モデル

カテゴリカル変数を目的変数としたとき、その条件付き確率を以下のように表現するものがロジスティック回帰モデルである。
$$
P(Y=1|X=\boldsymbol{x}) 
= \sigma(\beta_0 + \sum_{j=1}^J x_j \beta_j)
= \sigma(\boldsymbol{x}^\top\boldsymbol{\beta})
$$
ここで$\sigma$は**ロジスティック・シグモイド関数（logistic sigmoid function）**
$$
\sigma(z) = \frac{1}{1+\exp(-z)}
$$
という、$(-\infty, \infty)$の範囲の入力を$(0, 1)$の範囲に変換する関数である。



目的変数が二値である場合、つまり$y\in \{0, 1\}$の場合には、モデル全体は次のようになる
$$
\begin{align}
P(Y=y|X=\boldsymbol{x})
&= P(Y=1|X=\boldsymbol{x})^y P(Y=0|X=\boldsymbol{x})^{1-y}\\
&= \sigma(\boldsymbol{x}^\top\boldsymbol{\beta})^y
(1-\sigma(\boldsymbol{x}^\top\boldsymbol{\beta}))^{1-y}
\end{align}
$$


## パラメータの推定

目的変数と特徴量のサンプルが得られた下での尤度は
$$
P(\boldsymbol{y}|X) = \prod_{i=1}^n
\left[
\sigma(\boldsymbol{x}^\top\boldsymbol{\beta})^{y_i}
(1 - \sigma(\boldsymbol{x}^\top\boldsymbol{\beta}))^{1-y_i}
\right]
$$
で、負の対数尤度は
$$
-\log P(\boldsymbol{y}|X)
= - \sum_{i=1}^n
[
y_i \log \sigma(\boldsymbol{x}^\top\boldsymbol{\beta})
+
(1-y_i) \log(1 - \sigma(\boldsymbol{x}^\top\boldsymbol{\beta}))
]
$$
となる。この対数尤度をニュートン法で最小化する

