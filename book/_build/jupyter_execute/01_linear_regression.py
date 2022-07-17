#!/usr/bin/env python
# coding: utf-8

# # 線形回帰
# 
# ## 基本
# 
# ### モデル
# 
# **線形回帰**（linear regression）は、予測の目的変数$y$と特徴量（説明変数）$x_1, x_2, ..., x_d$の間に次のような線形関係を仮定したモデルを置いて予測する手法。
# $$
# y = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d + \varepsilon
# $$
# 
# ここで$\beta_1, \beta_2, ..., \beta_d$は回帰係数と呼ばれるパラメータで、モデル内で推定される。$\varepsilon$は誤差で、$\text{E}[\varepsilon]=0, \text{V}[\varepsilon]=\sigma^2$を仮定する。
# 
# 
# 
# サンプルサイズが$n$のデータセット$\{\boldsymbol{x}_i, y_i\}^n_{i=1}$があるとして、目的変数を$\boldsymbol{y} = (y_1, y_2, ..., y_n)^\top$、特徴量を$\boldsymbol{X}=(\boldsymbol{x}_1, \boldsymbol{x}_2, ..., \boldsymbol{x}_n)^\top$とおくと、このモデルは
# 
# $$
# \boldsymbol{y}=\boldsymbol{X} \boldsymbol{\beta}+\boldsymbol{\varepsilon}
# $$
# 
# と表記することができる。
# 
# 
# 
# ### パラメータの推定
# 
# 一般的に線形回帰ではパラメータの推定に**最小二乗法**（least squares method）という方法が使われる。
# 
# これは誤差関数$J( \boldsymbol{\beta})$を実測値$\boldsymbol{y}$と予測値$\hat{\boldsymbol{y}} = \boldsymbol{X}\hat{\boldsymbol{\beta}}$の二乗誤差の和（誤差二乗和 sum of squared error: SSE）
# 
# $$
# J( \boldsymbol{\beta}) = \sum^n_{i=1} (y_i - \hat{y}_i)^2 = \sum^n_{i=1} \varepsilon_i^2 = \boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}
# $$
# 
# として定義し、この二乗誤差を最小にするパラメータ（**最小二乗推定量** ordinary least square's estimator: OLSE）
# 
# $$
# \newcommand{\argmin}{\mathop{\rm arg~min}\limits}
# \hat{\boldsymbol{\beta}}^{LS} = \argmin_{\boldsymbol{\beta}}
# \sum^n_{i=1} (y_i - \hat{y}_i)^2
# $$
# 
# 
# を採用するという方法。
# 
# 二乗誤差$(y_i - \hat{y}_i)^2 = \varepsilon_i^2$は放物線を描くため、傾きがゼロになる点が最小値だと考えられる。そのため最小二乗法は解析的に解を求めることができる。
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

def square(error):
    return error ** 2

errors = np.linspace(-1, 1, 100)
squared_errors = [square(error) for error in errors]

fig, ax = plt.subplots()
ax.plot(errors, squared_errors)
_ = ax.set(xlabel='error', ylabel='squared error')


# 誤差二乗和は
# 
# $$
# \begin{align}
# \boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}
# &= (\boldsymbol{y} - \boldsymbol{X}\hat{\boldsymbol{\beta}})^\top (\boldsymbol{y} - \boldsymbol{X}\hat{\boldsymbol{\beta}})\\
# &=\boldsymbol{y}^\top \boldsymbol{y}
# - \boldsymbol{y}^\top\boldsymbol{X}\boldsymbol{\beta}
# - (\boldsymbol{X}\boldsymbol{\beta})^\top\boldsymbol{y}
# + (\boldsymbol{X}\boldsymbol{\beta})^\top (\boldsymbol{X}\boldsymbol{\beta})\\
# &= \boldsymbol{y}^\top \boldsymbol{y}
# - 2 \boldsymbol{\beta}^\top \boldsymbol{X}^\top \boldsymbol{y}
# + \boldsymbol{\beta}^\top \boldsymbol{X}^\top 
# \boldsymbol{X} \boldsymbol{\beta}\\
# \end{align}
# $$
# 
# ```{margin}
# ※転置の基本公式から、$(\boldsymbol{X}\hat{\boldsymbol{\beta}})^\top=\hat{\boldsymbol{\beta}}^\top\boldsymbol{X}^\top$、$\boldsymbol{y}^\top\boldsymbol{X}\hat{\boldsymbol{\beta}}=(\boldsymbol{X}\hat{\boldsymbol{\beta}})^\top\boldsymbol{y}=\hat{\boldsymbol{\beta}}^\top\boldsymbol{X}^\top\boldsymbol{y}$
# ```
# 
# であるから、二乗誤差の傾きがゼロになる点は
# 
# $$
# \frac{\partial \boldsymbol{\varepsilon}^\top \boldsymbol{\varepsilon}}{\partial \boldsymbol{\beta}}
# = -2\boldsymbol{X}^\top\boldsymbol{y}
# + 2(\boldsymbol{X}^\top\boldsymbol{X})\boldsymbol{\beta}
# =\boldsymbol{0}
# $$
# 
# と表すことができる。
# 
# これを整理して
# 
# $$
# 2(\boldsymbol{X}^\top\boldsymbol{X}) \boldsymbol{\beta}
# = 2\boldsymbol{X}^\top\boldsymbol{y}
# $$
# 
# これの両辺を2で割ると（あるいは誤差関数の定義の際に$1/2$を掛けておくと）、**正規方程式**（normal equation）とよばれる次の式が得られる。
# 
# $$
# (\boldsymbol{X}^\top\boldsymbol{X})\boldsymbol{\beta}
# = \boldsymbol{X}^\top\boldsymbol{y}
# $$
# 
# これを$\boldsymbol{\beta}$について解けば
# 
# $$
# \boldsymbol{\beta} =
# (\boldsymbol{X}^\top\boldsymbol{X})^{-1}
# \boldsymbol{X}^\top\boldsymbol{y}
# $$
# 
# となり、最小二乗推定量$\hat{\boldsymbol{\beta}}^{LS}$が得られる。
# 

# ## 実装
# 
# numpyでは、行列やベクトルの積は`@`という演算子で書くことができる。そのため、
# 
# ```python
# import numpy as np
# beta = np.linalg.inv(X.T @ X) @ X.T @ y
# ```
# 
# のように書けば上の式とおなじ演算を行うことができる。

# ### データの準備

# 乱数を発生させて架空のデータを作る。
# 
# $$
# y = 10 + 3 x_1 + 5 x_2 + \varepsilon\\
# x_1 \sim Uniform(0, 10)\\
# x_2 \sim Normal(3, 1)\\
# \varepsilon \sim Normal(0, 1)\\
# $$
# 
# ここで$\varepsilon$は測定誤差などのランダムなノイズとする

# In[2]:


import numpy as np
import pandas as pd
n = 100  # sample size

np.random.seed(0)
x0 = np.ones(shape=(n, ))
x1 = np.random.uniform(0, 10, size=n)
x2 = np.random.normal(3, 1, size=n)
noise = np.random.normal(size=n)

beta = [10, 3, 5]  # 真の回帰係数
y = beta[0] * x0 + beta[1] * x1 + beta[2] * x2 + noise 


# 特徴量$x$と目的変数$y$の関係を散布図で描くと次の図のようになった。

# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(ncols=2, figsize=(10, 3))
axes[0].scatter(x1, y)
axes[0].set(xlabel='x1', ylabel='y')
axes[1].scatter(x2, y)
axes[1].set(xlabel='x2', ylabel='y')


# ### 推定
# 
# これらのデータを使用して推定を行う。

# In[4]:


X = np.array([x0, x1, x2]).T

# Xの冒頭5行は以下のようになっている
print(X[0:5])


# In[5]:


# 最小二乗法で推定
beta_ = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"""
推定された回帰係数: {beta_.round(3)}
データ生成過程の係数: {beta}
""")


# 真の値にそれなりに近い回帰係数が推定できた。

# なお、scikit-learnに準拠したfit/predictのメソッドを持つ形でクラスとして定義するなら、以下のようになる（参考： [sklearn準拠モデルの作り方 - Qiita](https://qiita.com/roronya/items/fdf35d4f69ea62e1dd91)）。

# In[6]:


# scikit-learnに準拠した形で実装
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegression(BaseEstimator, RegressorMixin):

    def fit(self, X, y):
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        return X @ self.coef_


# In[7]:


model = LinearRegression()
model.fit(X, y)
model.coef_


# ### 予測してみる

# root mean squared error (RMSE)
# 
# $$
# RMSE = \sqrt{
#   \frac{1}{N} \sum^N_{i=1} (y_i - \hat{y}_i)^2
# }
# $$
# 
# を使って予測値を評価してみる。

# In[8]:


# 予測値を算出
y_pred = model.predict(X)

# 予測誤差を評価
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y, y_pred, squared=False)

print(f"RMSE: {rmse:.3f}")


# 予測値と実測値の散布図を描くと次のようになった。

# In[9]:


fig, ax = plt.subplots()
ax.plot(y_pred, y_pred, color='gray', alpha=.7)
ax.scatter(y_pred, y, alpha=.7)
_ = ax.set(xlabel='Predicted', ylabel='Actual')


# In[ ]:




