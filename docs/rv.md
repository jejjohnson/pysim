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

### Single Sample, Multiple Dimensions

Let $X \in \mathbb{R}^{1 \times D}$ and $Y \in \mathbb{R}^{1 \times D}$. This is the case where we have a single realization (single sample) 

$$\rho V(\mathbf{X}, \mathbf{Y})
=\frac{ tr\left( \Sigma_{XY} \right), \tilde{K}_\mathbf{y} \rangle_\mathcal{F}}{||\tilde{K}_\mathbf{x}||_\mathcal{F}||\tilde{K}_\mathbf{y}||_\mathcal{F}}$$


---

### 