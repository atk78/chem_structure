# カーン＝ヒリアード方程式

単一の均一混合物から2つの区別できる相分離の中でも**スピノーダル分解**（不安定状態から平衡状態への状態変化に対応する相分離構造）の時間発展の過程を記述する。高分子の混合物や合金のモデルとして用いられる。

流体の濃度$c$($c = \plusmn1$の領域で表す)としたとき、カーンヒリアード方程式は

$$
\frac{\partial c}{\partial t} = M \nabla^2\left[ \frac{\delta F}{\delta c}\right] = M \left[-\kappa \nabla^4 c + \nabla^2 f'(c)\right]
$$

$c$: 濃度、$t$: 時間、$M$: 移動度、$F$: 自由エネルギー、$f$: 内部自由エネルギー

$$
F[c] = \int \left[ \frac{\kappa}{2} (\nabla c(\boldsymbol{r} ))^2 + f(c)\right] \text{d}{\boldsymbol{r}}
$$

$$
f(c) = W c^2(1-c)^2
$$

濃度場をフーリエ級数展開で以下のように離散的に表す。

$$
c(\boldsymbol{r},t) = \frac{1}{L^2} \sum_{\boldsymbol{k}} \widehat{c}_{\boldsymbol{k}}(t) e^{i \boldsymbol{k} \cdot \boldsymbol{r} }
$$

ここでフーリエ係数は

$$
\widehat{c}_{\boldsymbol{k}}(t) = \mathcal{FT}[c(\boldsymbol{r},t) ] = \int_V  c(\boldsymbol{r},t)e^{-i \boldsymbol{k} \cdot \boldsymbol{r} }\text{d}{\boldsymbol{r}}
$$

であり、$k_i = \{-\pi N_i/L_i, -\pi(N_i-1)/L_i, \ldots, \pi(N_i-1)/L_i,\pi N_i/L_i\}$である。

ここで、$\Delta_i$は$i$方向のメッシュのグリッドサイズである。

定義式より

$$
\frac{\partial \widehat{c}_{\boldsymbol{k}} }{\partial t}   = M \left [ - k^2 \mathcal{FT}[f']-\kappa k^4 \widehat{c}_{\boldsymbol{k}} \right ]
$$

Euler陰解法より

$$
\frac{\widehat{c}_{\boldsymbol{k}}^{n+1} -\widehat{c}_{\boldsymbol{k}}^{n} }{\Delta t}=M\left [-k^2\mathcal{FT}[f'(c^n)]-\kappa k^4 \widehat{c}_{\boldsymbol{k}}^{n+1} \right ]
$$

と表すことができ、つまり、以下のように時間発展式とすることができる。

$$
\widehat{c}_{\boldsymbol{k}}^{n+1} =\frac{\widehat{c}_{\boldsymbol{k}}^n -\Delta t M k^2 \mathcal{FT}[f'(c^n)]}{1 +\Delta t \kappa k^4}
$$
