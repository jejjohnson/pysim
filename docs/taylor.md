### Taylor Diagram

The Taylor Diagram was a way to summarize the data statistics in a way that was easy to interpret. It used the relationship between the covariance, the correlation and the root mean squared error via the triangle inequality. Assuming we can draw a diagram using the law of cosines;

$$c^2 = a^2 + b^2 - 2ab \cos \phi$$

we can write this in terms of $\sigma$, $\rho$ and $RMSE$ as we have expressed above.

$$\text{RMSE}(X,Y)^2 = \sigma_{\text{obs}}^2 + \sigma_{\text{sim}}^2 - 2 \sigma_r \sigma_t \rho$$

The sides are as follows:

* $a = \sigma_{\text{obs}}$ - the standard deviation of the observed data
* $b = \sigma_{\text{sim}}$ - the standard deviation of the simulated data
* $\rho=\frac{C(X,Y)}{\sigma_x \sigma_y}$ - the correlation coefficient
* $RMSE$ - the root mean squared difference between the two datasets

So, the important quantities needed to be able to plot points on the Taylor diagram are the $\sigma$ and $\theta= \arccos \rho$. If we assume that the observed data is given by $\sigma_{\text{obs}}, \theta=0$, then we can plot the rest of the comparisons via $\sigma_{\text{sim}}, \theta=\arccos \rho$.

#### Example

<p float='center'> 
  <img src="thesis/appendix/information/pics/vi/demo_taylor.png" width="500" />
</p>

We see that the points are on top of each other. Makes sense seeing as how all of the other measures were also equivalent.
