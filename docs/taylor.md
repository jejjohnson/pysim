# Taylor Diagram

- [Taylor Diagram](#taylor-diagram)
  - [Motivation](#motivation)
    - [Questions](#questions)
    - [Current Ways](#current-ways)
  - [Cosine Similarity](#cosine-similarity)
  - [Law of Cosines](#law-of-cosines)
  - [Taylor Diagram](#taylor-diagram-1)
  - [Example](#example)
  - [Information Theory Diagram](#information-theory-diagram)

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

$$A \cdot B = ||A|| \; ||B|| \; cos \theta$$

$$
\begin{aligned}
\text{Similarity}
&= cos \theta \\
&= \frac{\mathbf{A\cdot B}}{\mathbf{||A||\;||B||}} \\
&= \frac{\sum_{i=1}^N A_i B_i}{\sqrt{\sum_{i=1}^N A_i^2} \sqrt{ \sum_{i=1}^N B_i^2}}
\end{aligned}$$


---

## Law of Cosines

The Taylor Diagram was a way to summarize the data statistics in a way that was easy to interpret. It used the relationship between the covariance, the correlation and the root mean squared error via the triangle inequality. Assuming we can draw a diagram using the law of cosines;

$$c^2 = a^2 + b^2 - 2ab \cos \phi$$


---

## Taylor Diagram

we can write this in terms of $\sigma$, $\rho$ and $RMSE$ as we have expressed above.

$$\text{RMSE}(X,Y)^2 = \sigma_{\text{obs}}^2 + \sigma_{\text{sim}}^2 - 2 \sigma_r \sigma_t \rho$$

The sides are as follows:

* $a = \sigma_{\text{obs}}$ - the standard deviation of the observed data
* $b = \sigma_{\text{sim}}$ - the standard deviation of the simulated data
* $\rho=\frac{C(X,Y)}{\sigma_x \sigma_y}$ - the correlation coefficient
* $RMSE$ - the root mean squared difference between the two datasets

So, the important quantities needed to be able to plot points on the Taylor diagram are the $\sigma$ and $\theta= \arccos \rho$. If we assume that the observed data is given by $\sigma_{\text{obs}}, \theta=0$, then we can plot the rest of the comparisons via $\sigma_{\text{sim}}, \theta=\arccos \rho$.

## Example

<p float='center'> 
  <img src="thesis/appendix/information/pics/vi/demo_taylor.png" width="500" />
</p>

We see that the points are on top of each other. Makes sense seeing as how all of the other measures were also equivalent.

---

## Information Theory Diagram