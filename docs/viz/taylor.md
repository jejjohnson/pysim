# Taylor Diagram

- [Motivation](#motivation)
  - [Questions](#questions)
  - [Current Ways](#current-ways)
- [Cosine Similarity](#cosine-similarity)
  - [Correlation](#correlation)
  - [Distances](#distances)
- [Taylor Diagram](#taylor-diagram)
    - [Law of Cosines](#law-of-cosines)
    - [Statistics Metric Space](#statistics-metric-space)
    - [Example 1 - Intuition](#example-1---intuition)
    - [Example 2 - Model Outputs](#example-2---model-outputs)
- [Multi-Dimensional Data](#multi-dimensional-data)
    - [Distances](#distances-1)
    - [Correlation](#correlation-1)
    - [Sample Space](#sample-space)
    - [Non-Linear Functions](#non-linear-functions)
- [Resources](#resources)

---

## Motivation

Visualizations:
* help find similarities between outputs
* stats are great, but visual uncertainty quantification

### Questions

* Which model is more **similar** to the reference/observations?
* Should we look at correlations across seasons or latitudes?
* Are there large discrepancies in the different outputs

### Current Ways

* **Trend Plots** often do not expose the comparison aspects...
* **Scatter** plots become impractical for many outputs
* **Parallel Coordinate** Plots are more practical, but only certain pairwise comparisons are possible
* **Plots per ensemble** - possible but it can be super cluttered
* **Taylor Diagram** - visualize several statistics simultaneously in a statistical metric space.

**Specific Statistics**

* Mean, Variance, Correlation
* Box Plots (and variations)


---

## Cosine Similarity

<center>
<p float='center'> 
  <img src="viz/pics/cosine_sim.png" width="500" />
</p>

**Figure I**: A visual representation of the cosine similarity.

</center>

The cosine similarity function measures the degree of similarity between two vectors.

$$A \cdot B = ||A|| \; ||B|| \; cos \theta$$

$$
\begin{aligned}
\text{sim}( x,y)
&= cos (\theta) = \frac{x\cdot y}{||x||\;||y||}
\end{aligned}$$

<details>
<summary><font color="red">Code</font></summary>


```python
def cosine_similarity(x: np.ndarray, y: np.ndarray) -> ~~float~~:
  """Computes the cosine similarity between two vectors X and Y
  Reflects the degree of similarity.

  Parameters
  ----------
  X : np.ndarray, (n_samples)

  Y : np.ndarray, (n_samples)

  Returns
  -------
  sim : float
    the cosine similarity between X and Y
  """
  # compute the dot product between two vectors
  dot = np.dot(x, y)

  # compute the L2 norm of x 
  x_norm = np.sqrt(np.sum(x ** 2))
  y_norm = np.linalg.norm(y)

  # compute the cosine similarity
  sim = dot / (x_norm * y_norm)
  return sim
```

</details>




### Correlation

There is a relationship between the cosine similarity and correlation coefficient

$$\rho(\mathbf{x}, \mathbf{y}) = \frac{ \text{cov}(\mathbf{x},\mathbf{y})}{\sigma_\mathbf{x} \sigma_\mathbf{y}}$$




---

### Distances


<center>
<p float='center'> 
  <img src="viz/pics/cosine_euclidean.png" width="500" />
</p>

**Figure II**: The triangle showing the cosine similarity and it's relationship to the euclidean distance.

</center>

$$d^2(x,y) = ||x-y||^2=||x||^2 + ||y||^2 - 2 \langle x, y \rangle$$

$$d^2(X,Y) = \sum_{i=1} \lambda_{x_i}^2 + \sum_{i=1} \lambda_{y_i}^2 - 2 \sum_{i,j=1} \lambda_{x_i}\lambda_{y_i}$$

**Correlation Coefficient**

$$\rho(X,Y) = \frac{\langle X,Y \rangle}{||X||\; ||Y||}$$

**Product of Scalars**

$$\begin{aligned}
\tilde{X} &= \frac{X}{||X||} \\
\tilde{Y} &= \frac{Y}{||Y||}
\end{aligned}
$$

* if $\rho(X,Y) = 0$, the spaces are orthogonal
* if $\rho(X,Y) = 1$, the spaces are equivalent, $d^2(X,Y) =0$





---

## Taylor Diagram

#### Law of Cosines

The Taylor Diagram was a way to summarize the data statistics in a way that was easy to interpret. It used the relationship between the covariance, the correlation and the root mean squared error via the triangle inequality. Assuming we can draw a diagram using the law of cosines;

$$c^2 = a^2 + b^2 - 2ab \cos \phi$$

#### Statistics Metric Space

we can write this in terms of $\sigma$, $\rho$ and RMSE as we have expressed above.

$$\text{RMSE}^2(x,y) = \sigma_{x}^2 + \sigma_{y}^2 - 2 \, \sigma_r \, \sigma_t \cos (\theta)$$

If we write out the full equation, we have the following:

$$\text{RMSE}^2(x,y) = \sigma_{x}^2 + \sigma_{y}^2 - 2 \, \sigma_r \, \sigma_t \, \rho (x,y)$$

The sides are as follows:

* $a = \sigma_{x}$ - the standard deviation of $x$
* $b = \sigma_{y}$ - the standard deviation of $y$
* $\rho=\frac{\text{cov}(x,y)}{\sigma_x \sigma_y}$ - the correlation coefficient
* RMSE - the root mean squared difference between the two datasets

So, the important quantities needed to be able to plot points on the Taylor diagram are the $\sigma$ and $\theta= \arccos \rho$. If we assume that the observed data is given by $\sigma_{\text{obs}}, \theta=0$, then we can plot the rest of the comparisons via $\sigma_{\text{sim}}, \theta=\arccos \rho$.

#### Example 1 - Intuition

<center>
<p float='center'> 
  <img src="viz/pics/taylor_demo.png" width="500" />
</p>

**Figure III**: An example Taylor diagram.

</center>


#### Example 2 - Model Outputs

## Multi-Dimensional Data

In the above examples, we assume that $\mathbf{x}, \mathbf{y}$ were both vectors of size $\mathbb{R}^{N \times 1}$. But what happens when we get datasets of size $\mathbb{R}^{N \times D}$? Well, the above formulas can generalize using the inner product and the norm of the datasets. 

<center>
<p float='center'> 
  <img src="viz/pics/cosine_vect.png" width="500" />
</p>

**Figure I**: A visual representation of the cosine similarity generalized to vectors.

</center>


#### Distances

We still get the same formulation as the above except now it is generalized to vectors.

$$d^2(\mathbf{x,y}) = ||\mathbf{x-y}||^2=||\mathbf{x}||^2 + ||\mathbf{y}||^2 - 2 \langle \mathbf{x,y} \rangle$$


#### Correlation

Let $\Sigma_\mathbf{xy}$ be the empirical covariance matrix between $\mathbf{x,y}$.



$$\rho V (\mathbf{x,y}) = \frac{\langle \Sigma_\mathbf{xy}, \Sigma_\mathbf{xy} \rangle_\mathbf{F}}{||\Sigma_\mathbf{xx}||_\mathbf{F} \; || \Sigma_\mathbf{yy}||_\mathbf{F}}$$

See the multidimensional section of this [page](linear/rv.md) for more details on the $\rho V$ coefficient.


#### Sample Space

Let $\mathbf{XX}^\top = \mathbf{W_x}$ and $\mathbf{YY}^\top = \mathbf{W_y}$

$$\rho V (\mathbf{x,y}) = \frac{\langle \mathbf{W_x}, \mathbf{W_y} \rangle_\mathbf{F}}{||\mathbf{W_x}||_\mathbf{F} \; || \mathbf{W_y}||_\mathbf{F}}$$

#### Non-Linear Functions

Let $\varphi(\mathbf{X}) = \mathbf{K_x}$ and $\varphi(\mathbf{Y}) = \mathbf{K_y}$. In the kernel community, this is known as the centered kernel alignment (cKA)

$$\text{cKA}(\mathbf{x,y}) = \frac{\langle \mathbf{K_x}, \mathbf{K_y} \rangle_\mathbf{F}}{||\mathbf{K_x}||_\mathbf{F} \; || \mathbf{K_y}||_\mathbf{F}}$$

---

## Resources

* [Le Traitement des Variables Vectorielles](https://www.jstor.org/stable/pdf/2529140.pdf?refreqid=excelsior%3Ad0e070c83ad4b47c30847094e65d99a7) - Yves Escoufier (1973)