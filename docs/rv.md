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


#### Summarizing

The only thing in the literature where I've seen this

**Observations**
* HSIC norm of the covariance only detects second order relationships. More complex (higher-order, nonlinear) relations cannot be captured

---

## Correlation

---

### Single Sample, Multiple Dimensions

Let $X \in \mathbb{R}^{1 \times D}$ and $Y \in \mathbb{R}^{1 \times D}$. This is the case where we have a single realization (single sample).

$$\rho V(\mathbf{X}, \mathbf{Y})
=\frac{ tr\left( \Sigma_{XY} \right), \tilde{K}_\mathbf{y} \rangle_\mathcal{F}}{||\tilde{K}_\mathbf{x}||_\mathcal{F}||\tilde{K}_\mathbf{y}||_\mathcal{F}}$$

**Code**

```python

```


---

### 