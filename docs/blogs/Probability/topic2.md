# Probability vs Likelihood

Probability and statistics have different meanings, though they are often misinterpreted or used interchangeably. Let's break it down and understand them further.

---

### **What is Probability?**

**Probability** is a branch of mathematics and statistics concerned with the numerical description of how likely events are to occur. The probability of an event is a number between **0 and 1** — the closer it is to 1, the more likely the event is to occur.


### **Example**

Tossing a fair (unbiased) coin. The outcomes are two — heads and tails. Since the coin is fair:

$$
P(\text{Head}) = 0.5,\quad P(\text{Tail}) = 0.5
$$

---


Now, let's say you toss the same fair coin 10 times and observe 7 heads and 3 tails. Let's compute the probability of this outcome.


This is a **Bernoulli experiment** repeated `n = 10` times, and follows the **Binomial distribution**. The probability of getting `k` successes (heads) is:

$$
P(\mathrm{k}) = \binom{n}{k} p^k (1 - p)^{n-k}
$$

Substituting in our values (`n = 10`, `k = 7`, `p = 0.5`):

$$
P(7 \text{ heads}) = \binom{10}{7} \cdot 0.5^7 \cdot 0.5^3 = 0.1172
$$


This is how you compute the probability **given** that the parameter (p = 0.5) is known.

---

### **What if the Coin is Biased?**

Now suppose you toss a **biased coin** 5 times and observe the outcomes:

- 3 heads  
- 2 tails  

But this time, you don’t know the coin’s probability p of landing heads. So instead of computing probability, we compute the likelihood of different values of p given the observed data.


- **Probability**: Given a model (parameter), how likely is the observed data?  
  Example:  
  $$
  P(\text{3 Heads} \mid p = 0.5)
  $$

- **Likelihood**: Given the observed data, how likely is a specific value of the parameter?  
  Example:  
 $$
  \mathcal{L}(p \mid \text{3 Heads})
 $$

$$
\mathcal{L}(p \mid \text{3 Heads}) = \binom{5}{3} \cdot p^3 \cdot (1 - p)^2
$$


We don't know p here, The value of \( p \) that **maximizes** this function is called the **Maximum Likelihood Estimate (MLE)**.

In our case, the MLE is:

$$
\hat{p} = 0.6,\quad \mathcal{L}(\hat{p}) = 0.3455
$$

<!-- <iframe src="C:\Users\abhij\PhD\DataBytes\docs\blogs\category1\likelihood_plot.html" width="100%" height="500px" frameborder="0"></iframe> -->

### Deriving the MLE for Likelihood

We are given the likelihood function (excluding the binomial coefficient since it does not affect the MLE):

$$
\mathcal{L}(p) = p^3 (1 - p)^2
$$


Let:

- \( u = p^3 \)
- \( v = (1 - p)^2 \)

Then the derivative using the product rule is:

$$
\frac{d{L}}{dp} = \frac{du}{dp} \cdot v + u \cdot \frac{dv}{dp}
$$



Now plug in:

$$
\frac{d{L}}{dp} = 3p^2(1 - p)^2 - 2p^3(1 - p)
$$

This can also be written equivalently as:

$$
\frac{d{L}}{dp} = p^2(p - 1)(5p - 3)
$$

Find Critical Points by setting the derivative to zero:

$$
p^2(1 - p)(3 - 5p) = 0
$$

Solutions:

- \( p = 0 \)
- \( p = 1 \)
- \( p = \frac{3}{5} \)

Select the Maximum, Since \( p = 0 \) and \( p = 1 \) yield zero likelihood, the maximum occurs at:

$$
\hat{p} = \frac{3}{5} = 0.6
$$

This is the **Maximum Likelihood Estimate (MLE)** for the parameter \( p \).

---

### In simpler terms:

- **Probability** evaluates **data given the parameter**
- **Likelihood** evaluates **parameter given the data**


| Feature              | Probability                                 | Likelihood                                       |
|----------------------|---------------------------------------------|--------------------------------------------------|
| **Viewpoint**        | Model → Data                                | Data → Model                                     |
| **Expression**       | \(P(X \mid θ)\)                             | \(\mathcal{L}(θ \mid X)\)                        |
| **Normalization**    | Integrates to 1 over *x*                    | Not normalized over *θ*                          |
| **Use-case**         | Predictions, simulations                    | Parameter inference (e.g. MLE)                   |


## Conclusion

- **Probability**: *Model → Data*.  
  *“Given θ, how likely is X?”*

- **Likelihood**: *Data → Model*.  
  *“Given observed X, how plausible is θ?”*

---
## References

- [Wikipedia: Likelihood function](https://en.wikipedia.org/wiki/Likelihood_function)  
- [Wikipedia: Probability](https://en.wikipedia.org/wiki/Probability)  





<!-- ## 6.  Gaussian (Normal) Distribution Likelihood

For a continuous random variable (e.g., height, weight) modeled by a **normal distribution**:

\[
f(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
\]

For data points \( x_1, x_2, \dots, x_n \), the **likelihood function** for \( \mu \) and \( \sigma \) is:

\[
\mathcal{L}(\mu, \sigma \mid \mathbf{x}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(x_i - \mu)^2}{2\sigma^2} \right)
\]

The **log-likelihood** is often used:

\[
\ell(\mu, \sigma) = -\frac{n}{2} \log(2\pi) - n \log \sigma - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2
\]

probability and statistics have different meanings. Often people misinterpret these. Let's break it down and understand it further.

Probability by definition: Defined as a branch of mathematics and statistics concerning about events and numerical descriptions of how likely they are to occur. The probability of an event is a number betweeen 0 and 1. The larger the value more likely an event to occur. 

Ex: Tossing a fair (unbiased) coin, the outcomes are two since the coin is fair (heads and tails), So the probability of seeing either head or tail is equal and is 0.5.

Let's take the example of the same coin, however now you toss the coin multiple times and observed the outcomes.

A coin is tossed 10 times and noticed 7 heads, 3 tails. Now let's calculate the probability of seeing 7 heads and we know the coin is fair (p=0.5).

This can be consider as a bernoulli experiment repeated n ( here 10 ) times. The probability of seeing k successes is (n k) ( p)^k (1-p)^1-k

probability(k=7)=

What if the coin is not fair and since it's biased we can't say the probaility of both outcomes are same.

So you tossed a biased coin and the observed the following data.

3 heads and 2 tails and here we don't know the p- value.

So in better terms:

probability is how likely the event is gonna occur or how well our model or hypothesis is explains the observed data. .i.e. P(7 Heads|p=0.5)

Where as likelihood is given the observed data how likely is your parameters.

L(p| 3 Heads) -->