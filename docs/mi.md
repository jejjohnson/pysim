# Mutual Information

> How much information one random variable says about another random variable.

- [Intiution](#intiution)
- [Full Definition](#full-definition)
- [Code](#code)
- [Supplementary](#supplementary)
  - [Information](#information)
    - [Intuition](#intuition)
    - [Formulation](#formulation)
    - [Units](#units)
  - [Entropy](#entropy)
    - [Intuition](#intuition-1)
    - [Single Variable](#single-variable)
    - [Code - Step-by-Step](#code---step-by-step)
    - [Multivariate](#multivariate)
    - [Relative Entropy (KL-Divergence)](#relative-entropy-kl-divergence)


---

## Intiution

* Measure of the amount of information that one RV contains about another RV
* Reduction in the uncertainty of one rv due to knowledge of another
* The intersection of information in X with information in Y

---

## Full Definition

$$I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

$$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

**Sources**:
* [Scholarpedia](http://www.scholarpedia.org/article/Mutual_information)

---

## Code

1. We need a PDF estimation...


2. Normalize counts to probability values

```python
pxy = bin_counts / float(np.sum(bin_counts))
```

3. Get the marginal distributions

```python
px = np.sum(pxy, axis=1) # marginal for x over y
py = np.sum(pxy, axis=0) # marginal for y over x
```

4. Joint Probability

---

## Supplementary

### Information


#### Intuition

> Things that don't normally happen, happen.


#### Formulation

$$I(X) = - \log \frac{1}{p(X)}$$


#### Units

<center>

| Base  | Units |      Conversion      |        Approximate         |
| :---: | :---: | :------------------: | :------------------------: |
|   2   | bits  |    1 bit = 1 bit     |       1 bit = 1 bit        |
|   e   | nats  |  1 bit = $\log_e 2$  | 1 bit $\approx$ 0.693 nats |
|  10   | bans  | 1 bit = $\log_{10}2$ | 1 bit $\approx$ 0.301 bans |

</center>

---

### Entropy


#### Intuition

> Expected uncertainty.

$$H(X) = \log \frac{\text{\# of Outcomes}}{\text{States}}$$

* Lower bound on the number of bits needed to represent a RV, e.g. a RV that has a unform distribution over 32 outcomes.
  * Lower bound on the average length of the shortest description of $X$
* Self-Information

#### Single Variable

$$H(X) = \mathbb{E}_{p(X)} \left( \log \frac{1}{p(X)}\right)$$


<details>
<summary>Code - From Scratch</summary>

#### Code - Step-by-Step

1. Obtain all of the possible occurrences of the outcomes. 
   ```python
   values, counts = np.unique(labels, return_counts=True)
   ```

2. Normalize the occurrences to obtain a probability distribution
   ```python
   counts /= counts.sum()
   ```

3. Calculate the entropy using the formula above
   ```python
   H = - (counts * np.log(counts, 2)).sum()
   ```

As a general rule-of-thumb, I never try to reinvent the wheel so I look to use whatever other software is available for calculating entropy. The simplest I have found is from `scipy` which has an entropy function. We still need a probability distribution (the counts variable). From there we can just use the entropy function.
</details>


<details>
<summary>Code - Refactor</summary>

2. Use Scipy Function
   ```python
   H = entropy(counts, base=base)
   ```
</details>

#### Multivariate

$$H(X) = \mathbb{E}_{p(X,Y)} \left( \log \frac{1}{p(X,Y)}\right)$$

#### Relative Entropy (KL-Divergence)

Measure of distance between two distributions

$$D_{KL} (P,Q) = \int_\mathcal{X} p(x) \:\log \frac{p(x)}{q(x)}\;dx$$

* aka expected log-likelihood ratio
* measure of inefficiency of assuming that the distribution is $q$ when we know the true distribution is $p$.

