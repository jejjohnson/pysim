# RV Coefficient

**Notation**

* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{x}}$ are samples from a multidimentionsal r.v. $\mathcal{X}$
* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{y}}$ are samples from a multidimensional r.v. $\mathcal{Y}$
* $\Sigma \in \mathbb{R}^{N \times N}$ is a covariance matrix.
  * $\Sigma_\mathbf{x}$ is a kernel matrix for the r.v. $\mathcal{X}$
  * $\Sigma_\mathbf{y}$ is a kernel matrix for the r.v. $\mathcal{Y}$
  * $\Sigma_\mathbf{xy}$ is the population covariance matrix between $\mathcal{X,Y}$
* $tr(\cdot)$ - the trace operator
* $||\cdot||_\mathcal{F}$ - Frobenius Norm
  * $||\cdot||_\mathcal{HS}$ - Hilbert-Schmidt Norm 
* $\tilde{K} \in \mathbb{R}^{N \times N}$ is the centered kernel matrix.

---

## Covariance

$$C(X,Y)=\mathbb{E}\left((X-\mu_x)(Y-\mu_y) \right)$$

<details>
<summary>
    <font color="black">Alternative Formulations
    </font>
</summary>

$$
\begin{aligned}
C(X,Y) &= \mathbb{E}\left((X-\mu_x)(Y-\mu_y) \right) \\
&= \mathbb{E}\left(XY - \mu_xY - X\mu_y + \mu_x\mu_y \right) \\
&=  \mathbb{E}(XY) - \mu_x  \mathbb{E}(X) -  \mathbb{E}(X)\mu_y + \mu_x\mu_y \\
&=  \mathbb{E}(XY) - \mu_x\mu_y
\end{aligned}
$$
</details>

* Measures Dependence
* Unbounded, $(-\infty,\infty)$
* Isotropic scaling
* Units depend on inputs

### Empirical Covariance

This shows the joint variation of all pairs of random variables.

$$C_{xy} = X^\top X$$


<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ X
```
</details>

**Observations**
* A completely diagonal covariance matrix means that all features are uncorrelated (orthogonal to each other).
* Diagonal covariances are useful for learning, they mean non-redundant features!


### Empirical Cross-Covariance 

This is the covariance between different datasets

$$C_{xy} = X^\top Y$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ Y
```
</details>

### Linear Kernel 

This measures the covariance between samples.

$$K_{xx} = X X^\top$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
K_xy = X @ X.T
```
</details>

* A completely diagonal linear kernel (Gram) matrix means that all examples are uncorrelated (orthogonal to each other).
* Diagonal kernels are useless for learning: no structure found in the data.

---

### Hilbert-Schmidt Norm


We can also consider the case where the correlations can be measured between samples and not between features. So we can create cross product matrices: $\mathbf{W}_\mathbf{X}=\mathbf{XX}^\top \in \mathbb{R}^{N \times N}$ and $\mathbf{W}_\mathbf{Y}=\mathbf{YY}^\top \in \mathbb{R}^{N \times N}$. To measure the proximity, we can use the Hilbert-Schmidt (HS) norm, $||\cdot||_{F}$. 

$$
\begin{aligned}
\langle {W}_\mathbf{X}, {W}_\mathbf{Y} \rangle 
&= 
tr \left( \mathbf{XX}^\top \mathbf{YY}^\top \right) \\
&= 
\sum_{i=1}^{D_x} \sum_{j=1}^{D_y} cov^2(\mathbf{X}_{d_i}, \mathbf{Y}_{d_j})
\end{aligned}
$$

**Observations**
* HSIC norm of the covariance only detects second order relationships. More complex (higher-order, nonlinear) relations cannot be captured

---

## Correlation

---

### Correlation Coefficient - $\rho$

$$\rho(X, Y) = \frac{C(X,Y)}{\sigma_x \sigma_y}$$

---

### RV Coefficient - $\rho$V

Let $X \in \mathbb{R}^{1 \times D}$ and $Y \in \mathbb{R}^{1 \times D}$. This is the case where we have a single realization (single sample).

$$
\begin{aligned}
\rho V(X,Y)
&=
\frac{ tr\left( \Sigma_{XY}\Sigma_{XY}^\top \right)}{\sqrt{tr\left( \Sigma_{XX}^2  \right) tr\left( \Sigma_{YY}^2  \right) }} \\
&= \frac{tr\left( \Sigma_{XY}\Sigma_{XY}^\top \right)}{||\Sigma_{XX}||\;||\Sigma_{YY}||}
\end{aligned}
$$

### RV Coefficient - RV (feature/primal space, $\mathbb{R}^D$)

Let's add $N$ independent realizations to the samples. This gives us a vector for each of the observations. So, let $\mathbf{X} \in \mathbb{R}^{N \times D_x}$ and $\mathbf{Y} \in \mathbb{R}^{N \times D_y}$. We assume that they are column-centered (aka remove the mean from the features). So, we can write the $S_{\mathbf{XY}}= \frac{1}{n-1}\mathbf{X^\top Y}$

$$
\begin{aligned}
\text{RV}(\mathbf{X,Y})
&= 
\frac{tr\left( S_{\mathbf{XY}}S_{\mathbf{XY}} \right)}{\sqrt{tr\left( S_{\mathbf{XX}}^2 \right) tr\left( S_{\mathbf{YY}}^2 \right)}}
\end{aligned}
$$

### RV Coefficient - RV (samples/dual space, $\mathbb{R}^N$)

$$
\begin{aligned}
\text{RV}(\mathbf{X,Y}) 
&=
\frac{\langle \mathbf{W_X, W_Y}\rangle}{||\mathbf{W_X}|| \; ||\mathbf{W_Y}||} \\
&=
\frac{tr\left( \mathbf{XX}^\top \mathbf{YY}^\top \right)}{\sqrt{tr\left( \mathbf{XX}^\top \right)^2 tr\left( \mathbf{XX}^\top \right)^2}}
\end{aligned}
$$


---

### 