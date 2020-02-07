# RV Coefficient

* Author: J. Emmanuel Johnson
* Email: jemanjohnson34@gmail.com
* Lab: [Colab Notebook](https://colab.research.google.com/drive/19bJd_KNTSThZcxP1vnQOVjTLTOFLS9VG)

---

- [Notation](#notation)
- [Single Variables](#single-variables)
  - [Mean, Expectation](#mean-expectation)
    - [Empirical Estimate](#empirical-estimate)
  - [Variance, $\mathbb{R}$](#variance-mathsemanticsmrowmi-mathvariant%22double-struck%22rmimrowannotation-encoding%22applicationx-tex%22mathbbrannotationsemanticsmathr)
    - [Empirical Estimate](#empirical-estimate-1)
  - [Covariance](#covariance)
    - [Empirical Estimate](#empirical-estimate-2)
  - [Correlation](#correlation)
    - [Empirical Estimate](#empirical-estimate-3)
- [Multi-Dimensional](#multi-dimensional)
- [Covariance](#covariance-1)
  - [Single Variable](#single-variable)
    - [Empirical Estimation](#empirical-estimation)
  - [Multivariate Covariance](#multivariate-covariance)
  - [Empirical Cross-Covariance](#empirical-cross-covariance)
  - [Linear Kernel](#linear-kernel)
  - [Summarizing Multi-Dimensional Information](#summarizing-multi-dimensional-information)
  - [Hilbert-Schmidt Norm](#hilbert-schmidt-norm)
  - [RV Coefficient - $\rho$V](#rv-coefficient---mathsemanticsmrowmi%cf%81mimrowannotation-encoding%22applicationx-tex%22rhoannotationsemanticsmath%cf%81v)
  - [RV Coefficient - RV (feature/primal space, $\mathbb{R}^D$)](#rv-coefficient---rv-featureprimal-space-mathsemanticsmrowmsupmi-mathvariant%22double-struck%22rmimidmimsupmrowannotation-encoding%22applicationx-tex%22mathbbrdannotationsemanticsmathrd)
  - [RV Coefficient - RV (samples/dual space, $\mathbb{R}^N$)](#rv-coefficient---rv-samplesdual-space-mathsemanticsmrowmsupmi-mathvariant%22double-struck%22rmiminmimsupmrowannotation-encoding%22applicationx-tex%22mathbbrnannotationsemanticsmathrn)
  - [](#)
- [Supplementary](#supplementary)

---

## Notation

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

## Single Variables

Let's consider a single variable $X \in \mathbb{R}^{N \times 1}$ which represents a set of samples of a single feature. 

* Mean
* Variance
* Covariance
* Correlation
* Visualize
  * Scatter Plots
  * Taylor Diagram
* Root Mean Squared Error

### Mean, Expectation

The first order measurement is the mean. This is the expected/average value that we would expect from a r.v.. This results in a scalar value


#### Empirical Estimate

$$\mu(x)=\frac{1}{N}\sum_{i=1}x_i$$

### Variance, $\mathbb{R}$

The first measure we need to consider is the variance. This is a measure of spreadcan be used for a 


#### Empirical Estimate


$$
\begin{aligned}
\sigma_x^2 
&= \frac{1}{n-1} \sum_{i=1}^N(x_i-x_\mu)(x_i-x_\mu) \\
&= \frac{1}{n-1} \sum_{i=1}^N(x_i-x_\mu)^2
\end{aligned}
$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

```python
# remove mean from data
X_mu = X.mean(axis=0)

# ensure it is 1D
var = (X - X_mu[:, None]).T @ (X - X_mu[:, None])
```

</details>

---

### Covariance


The first measure we need to consider is the covariance. This can be used for a single variable $X \in \mathbb{R}^{N \times 1}$ which represents a set of samples of a single feature. We can compare the r.v. $X$ with another r.v. $Y \in \mathbb{R}^{N \times 1}$. the covariance, or the cross-covariance between multiple variables $X,Y$. This results in a scalar value , $\mathbb{R}$. We can write this as:

$$
\begin{aligned}
C_{XY}(X,Y) &= \mathbb{E}\left[(X-\mu_x)(Y-\mu_y) \right] \\
&= \mathbb{E}[XY] - \mu_X\mu_Y
\end{aligned}
$$

<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

$$
\begin{aligned}
C(X,Y) &= \mathbb{E}\left((X-\mu_x)(Y-\mu_y) \right) \\
&= \mathbb{E}\left(XY - \mu_xY - X\mu_y + \mu_x\mu_y \right) \\
&=  \mathbb{E}(XY) - \mu_x  \mathbb{E}(X) -  \mathbb{E}(X)\mu_y + \mu_x\mu_y \\
&=  \mathbb{E}(XY) - \mu_x\mu_y
\end{aligned}
$$
</details>

#### Empirical Estimate

We can compare the r.v. $X$ with another r.v. $Y \in \mathbb{R}^{N \times 1}$. the covariance, or the cross-covariance between multiple variables $X,Y$. We can write this as:

$$\sigma(x,y) = \frac{1}{n-1} \sum_{i=1}^N (x_i - x_\mu)(y_i - y_\mu)$$

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

```python
c_xy = X.T @ Y
```
</details>

---

### Correlation


This results in a scalar value $\mathbb{R}$.

$$\rho(x,y)=\frac{\sigma(x,y)}{\sigma_x \sigma_y}$$

#### Empirical Estimate


---

## Multi-Dimensional

* Mean
* Variance, Covariance
* Correlation (RV Coefficient)
* Summarizing Information (HS Norm)
* Visualization
  * Scatter Plot
  * Correlation Plot
  * Taylor Diagram

---

## Covariance

---

### Single Variable

The first measure we need to consider is the covariance. This can be used for a single variable $X \in \mathbb{R}^{N \times 1}$ which represents a set of samples of a single feature. We can compare the r.v. $X$ with another r.v. $Y \in \mathbb{R}^{N \times 1}$. the covariance, or the cross-covariance between multiple variables $X,Y$. We can write this as:

$$
\begin{aligned}
C_{XY}(X,Y) &= \mathbb{E}\left[(X-\mu_x)(Y-\mu_y) \right] \\
&= \mathbb{E}[XY] - \mu_X\mu_Y
\end{aligned}
$$

<details>
<summary>
    <font color="red">Proof
    </font>
</summary>

We can expand the terms in the parenthesis like normally. Then we take the expectation of each of the terms individually.

$$
\begin{aligned}
C(X,Y) &= \mathbb{E}\left((X-\mu_x)(Y-\mu_y) \right) \\
&= \mathbb{E}\left(XY - \mu_xY - X\mu_y + \mu_x\mu_y \right) \\
&=  \mathbb{E}(XY) - \mu_x  \mathbb{E}(X) -  \mathbb{E}(X)\mu_y + \mu_x\mu_y \\
&=  \mathbb{E}(XY) - \mu_x\mu_y
\end{aligned}
$$
</details>

This results in a scalar value which represents the similarity between the samples. There are some key observations of this measure.

* When $\in \mathbb{R}^{N \times 1}$, the result is a scalar value.
* It is a measure of the joint variability between the datasets
* It is difficult to interpret because it can range from $-\infty$ to $\infty$. 
* The units are dependent upon the inputs. 
* It is affected by isotropic scaling

#### Empirical Estimation

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

---

### Multivariate Covariance

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


---
### Summarizing Multi-Dimensional Information

Let's have the two distributions $\mathcal{X} \in \mathbb{R}^{D_x}$ and $\mathcal{Y} \in \mathbb{R}^{D_y}$. Let's also assume that we can sample $(x,y)$ from $\mathbb{P}_{xy}$. We can capture the second order dependencies between $X$ and $Y$ by constructing a covariance matrix in the feature space defined as:

$$C_{\mathbf{xy}} \in \mathbb{R}^{D \times D}$$

We can use the Hilbert-Schmidt Norm (HS-Norm) as a statistic to effectively summarize content within this covariance matrix. It's defined as:

$$||C_{xy}||_{\mathcal{F}}^2 = \sum_i \lambda_i^2 = \text{tr}\left[ C_{xy}^\top C_{xy} \right]$$
 
 Note that this term is zero iff $X$ and $Y$ are independent and greater than zero otherwise. Since the covariance matrix is a second-order measure of the relations, we can only summarize the the second order relation information. But at the very least, we now have a scalar value that summarizes the structure of our data.

 

<details>
<summary>
    <font color="blue">Code
    </font>
</summary>

This is very easy to compute in practice. One just needs to calculate the Frobenius Norm (Hilbert-Schmidt Norm) of a covariance matrix This boils down to computing the trace of the matrix multiplication of two matrices: $tr(C_{xy}^\top C_{xy})$. So in algorithmically that is:

```python
hsic_score = np.sqrt(np.trace(C_xy.T * C_xy))
```
We can make this faster by using the `sum` operation

```python
# Numpy
hsic_score = np.sqrt(np.sum(C_xy * C_xy))
# PyTorch
hsic_score = (C_xy * C_xy).sum().sum()
```

**Refactor**

There is a built-in function to be able to to speed up this calculation by a magnitude.

```python
hs_score = np.linalg.norm(C_xy, ord='fro')
```

and in PyTorch

```python
hs_score = torch.norm(C_xy, p='fro)
```
</details>

And also just like the correlation, we can also do a normalization scheme that allows us to have an interpretable scalar value. This is similar to the correlation coefficient except it can now be applied to multi-dimensional data.

$$\rho_\mathbf{xy} = \frac{ ||C_{\mathbf{xy}}||_\mathcal{F}^2}{||C_\mathbf{xx}||_{\mathcal{F}} ||C_\mathbf{yy}||_{\mathcal{F}}}$$


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


---

## Supplementary

* Common Statistical Tests are Linear Models (or: How to Teach Stats) - Jonas Kristoffer Lindelov - [notebook](https://eigenfoo.xyz/tests-as-linear/) | [rmarkdown](https://lindeloev.github.io/tests-as-linear/)
* Correlation vs Regression - Asim Jana - [blog](https://www.datasciencecentral.com/profiles/blogs/difference-between-correlation-and-regression-in-statistics)
* RealPython
  * Numpy, SciPy and Pandas: Correlation with Python - [blog](https://realpython.com/numpy-scipy-pandas-correlation-python/)
* Correlation and Lag for Signals - [notebook](https://currents.soest.hawaii.edu/ocn_data_analysis/_static/SEM_EDOF.html)
* [Understanding the Covariance Matrix](https://datascienceplus.com/understanding-the-covariance-matrix/)