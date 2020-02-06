# Kernel Measures of Similarity


**Notation**

* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{x}}$ are samples from a multidimentionsal r.v. $\mathcal{X}$
* $\mathbf{X} \in \mathbb{R}^{N \times D_\mathbf{y}}$ are samples from a multidimensional r.v. $\mathcal{Y}$
* $K \in \mathbb{R}^{N \times N}$ is a kernel matrix.
  * $K_\mathbf{x}$ is a kernel matrix for the r.v. $\mathcal{X}$
  * $K_\mathbf{y}$ is a kernel matrix for the r.v. $\mathcal{Y}$
  * $K_\mathbf{xy}$ is the cross kernel matrix for the r.v. $\mathcal{X,Y}$
* $\tilde{K} \in \mathbb{R}^{N \times N}$ is the centered kernel matrix.

**Observations**

* $\mathbf{X},\mathbf{Y}$ can have different number of dimensions
* $\mathbf{X},\mathbf{Y}$ must have different number of samples

---
- [Kernel Measures of Similarity](#kernel-measures-of-similarity)
  - [Covariance Measures](#covariance-measures)
    - [Uncentered Kernel](#uncentered-kernel)
    - [Centered Kernel](#centered-kernel)
      - [Hilbert-Schmidt Independence Criterion (HSIC)](#hilbert-schmidt-independence-criterion-hsic)
      - [Maximum Mean Discrepency (MMD)](#maximum-mean-discrepency-mmd)
  - [Correlation Measures](#correlation-measures)
    - [Uncentered Kernel](#uncentered-kernel-1)
      - [Kernel Alignment (KA)](#kernel-alignment-ka)
    - [Uncentered Kernel](#uncentered-kernel-2)
      - [Centered Kernel Alignment (cKA)](#centered-kernel-alignment-cka)


---

## Covariance Measures

### Uncentered Kernel

 $$\text{cov}(\mathbf{X}, \mathbf{Y}) =||K_{\mathbf{xy}}||_\mathcal{F}
=\langle K_\mathbf{x}, K_\mathbf{y} \rangle_\mathcal{F}$$

---

### Centered Kernel

---

#### Hilbert-Schmidt Independence Criterion (HSIC)

$$\text{cov}(\mathbf{X}, \mathbf{Y}) =||\tilde{K}_{\mathbf{xy}}||_\mathcal{F}
=\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}$$

---

#### Maximum Mean Discrepency (MMD)


$$\text{cov}(\mathbf{X}, \mathbf{Y}) = ||K_\mathbf{x}||_\mathcal{F} + ||K_\mathbf{y}||_\mathcal{F}  -  \langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}$$

**[Source](https://github.com/choasma/HSIC-bottleneck/blob/master/source/hsicbt/math/hsic.py#L69)**




---

## Correlation Measures

---

### Uncentered Kernel

---

#### Kernel Alignment (KA)

$$\rho(\mathbf{X}, \mathbf{Y})
=\frac{\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}}{||\tilde{K}_\mathbf{x}||_\mathcal{F}||\tilde{K}_\mathbf{y}||_\mathcal{F}}$$

**In the Literature**

* Kernel Alignment

---

### Uncentered Kernel

---

#### Centered Kernel Alignment (cKA)

 $$\rho(\mathbf{X}, \mathbf{Y})
=\frac{\langle \tilde{K}_\mathbf{x}, \tilde{K}_\mathbf{y} \rangle_\mathcal{F}}{||\tilde{K}_\mathbf{x}||_\mathcal{F}||\tilde{K}_\mathbf{y}||_\mathcal{F}}$$


**In the Literature**

* Centered Kernel Alignment
