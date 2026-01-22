# **Probability vs. Likelihood: What’s the Difference?**

In statistics and data science, **probability** and **likelihood** are related but distinct concepts. This guide breaks down their difference, provides formulas, and includes practical Python examples.

---

**## 1. Definitions**

### Probability

- **Probability** quantifies how likely a **future event** (data) is, _given a known model or hypothesis_.
- Mathematically:
  $$
  P(X = x \mid \theta)
  $$
- Ranges from 0 to 1.
- Example: The chance of getting 2 heads in 3 fair coin flips:
  $$
  P(X=2 \mid p=0.5) = \binom{3}{2} 0.5^2 0.5^1 = 0.375
  $$

---

### Likelihood

- **Likelihood** measures how plausible a **parameter** (θ) is, _given observed data_.
- Treat data as fixed and view the model/hypothesis as variable.
- Defined as:
  $$
  \mathcal{L}(\theta \mid x) = P(x \mid \theta) \quad \text{(for discrete)}
  $$
  or
  $$
  \mathcal{L}(\theta \mid x) = f(x \mid \theta) \quad \text{(for continuous)}
  $$
- Crucially, likelihood is **not** a probability distribution over θ (does not integrate to 1).

---

## 2. Inferential Direction

- **Probability**: _Model → Data_.  
  _“Given θ, how likely is X?”_

- **Likelihood**: _Data → Model_.  
  _“Given observed X, how plausible is θ?”_

---

## 3. Example: Coin Flip

Imagine we toss a coin 10 times, observing 7 heads (x = 7).

- As **Probability**:
  $$
  P(X = 7 \mid p) = \binom{10}{7} p^7 (1-p)^3
  $$
- As **Likelihood**:
  $$
  \mathcal{L}(p \mid x=7) = P(X = 7 \mid p)
  $$
  This function of _p_ tells us which values are most plausible.

Often, we maximize it (_Maximum Likelihood Estimate_, MLE):

$$
\hat{p}\_{\text{MLE}} = \underset{p}{\arg\max}\;\mathcal{L}(p \mid x)
$$

For binomial data, this gives:

$$
\hat{p} = \frac{x}{n} = \frac{7}{10} = 0.7
$$

---

## 4. Distinguishing Features

| Feature           | Probability              | Likelihood                     |
| ----------------- | ------------------------ | ------------------------------ |
| **Viewpoint**     | Model → Data             | Data → Model                   |
| **Expression**    | \(P(X \mid θ)\)          | \(\mathcal{L}(θ \mid X)\)      |
| **Normalization** | Integrates to 1 over _x_ | Not normalized over _θ_        |
| **Use-case**      | Predictions, simulations | Parameter inference (e.g. MLE) |

---

## 5. **Python Example**

```python
import numpy as np
from scipy.stats import binom

# Observed data
n, x_obs = 10, 7
ps = np.linspace(0, 1, 101)

# Likelihood values
likelihoods = binom.pmf(x_obs, n, ps)

# MLE
p_mle = ps[np.argmax(likelihoods)]

print(f"Observed {x_obs}/{n} heads.")
print(f"MLE for p: {p_mle:.2f}")  # → 0.70
```

---

## 6. Gaussian (Normal) Distribution Likelihood

For a continuous random variable (e.g., height, weight) modeled by a **normal distribution**:

$$
f(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

For data points \( x_1, x_2, \dots, x_n \), the **likelihood function** for \( \mu \) and \( \sigma \) is:

$$
\mathcal{L}(\mu, \sigma \mid \mathbf{x}) = \prod\_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right)
$$

The **log-likelihood** is often used:

$$
\ell(\mu, \sigma) = -\frac{n}{2} \log(2\pi) - n \log \sigma - \frac{1}{2\sigma^2} \sum\_{i=1}^n (x_i - \mu)^2
$$

### Python Example for Gaussian Likelihood

```python
import numpy as np
from scipy.stats import norm

# Observed data (example)
data = np.array([5.1, 4.9, 5.0, 5.2, 4.8])

# Range of mu values to evaluate likelihood
mus = np.linspace(4.5, 5.5, 100)
sigma = 0.1

# Compute likelihood for each mu
likelihoods = np.array([np.prod(norm.pdf(data, loc=mu, scale=sigma)) for mu in mus])

# MLE for mu
mu_mle = mus[np.argmax(likelihoods)]

print(f"MLE for mu (mean): {mu_mle:.2f}")
```

---

## 7. Summary

- **Probability**: From model to data; predicts outcomes _given_ parameters.
- **Likelihood**: From data to model; infers parameters _given_ observed outcomes.
- Though both use the same mathematical function, the **variable of interest changes**.

---

## 8. References

- [Wikipedia: Likelihood function](https://en.wikipedia.org/wiki/Likelihood_function)
- [Wikipedia: Probability](https://en.wikipedia.org/wiki/Probability)
- Stats.SE clarification: "Likelihood is … probability of observed outcomes given parameters"
- Articles from Built In, GeeksforGeeks and other statistics blogs.
