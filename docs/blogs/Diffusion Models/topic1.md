# Diffusion Models

Diffusion models look magical “start from noise, end with an image”, but the math is a clean chain:

probability & density → likelihood → KL divergence → ELBO → diffusion as a latent-variable model → a simple MSE loss (predicting noise)

---

## Table of contents

- [1. Probability vs density](#1-probability-vs-density)
- [2. Likelihood and why it becomes a product](#2-likelihood-and-why-it-becomes-a-product)
- [3. Conditional probability, chain rule, and the Markov property](#3-conditional-probability-chain-rule-and-the-markov-property)
- [4. KL divergence (what it measures and where it comes from)](#4-kl-divergence-what-it-measures-and-where-it-comes-from)
- [5. ELBO (the “evidence lower bound”) from scratch](#5-elbo-the-evidence-lower-bound-from-scratch)
- [6. Diffusion models as a latent-variable model](#6-diffusion-models-as-a-latent-variable-model)
- [7. From diffusion ELBO to the simple MSE loss](#7-from-diffusion-elbo-to-the-simple-mse-loss)
- [8. Sampling: the reverse process](#8-sampling-the-reverse-process)
- [9. Next: a 2D toy diffusion model you can _see_](#9-next-a-2d-toy-diffusion-model-you-can-see)
- [References](#references)

---

## Notation

- Random variables: \(X, Z\). Observed values: \(x, z\).
- Probability mass function (discrete): \(p(x)\).
- Probability density function (continuous): \(p(x)\) (same letter; context tells you which).
- Expectation: \(\mathbb{E}[\cdot]\).
- KL divergence: $D_{KL}(P \| Q)$

---

## 1. Probability vs density

### Discrete case (PMF)

If \(X\) is discrete (like a dice roll), then:

- \(p(X=3)\) is an honest probability number.
- \(\sum_x p(x) = 1\).

### Continuous case (PDF)

If \(X\) is continuous (like a real number), then:

- \(p(X = 0.5) = 0\) (exact points have zero probability mass),
- but we can talk about probability of **intervals**:
  $$
    P(a \le X \le b) = \int_a^b p(x)\,dx.
  $$

So what is \(p(x)\) in the continuous case?

- It’s a **density**: it tells you how “packed” or "dense" probability is near \(x\).
- A density can be bigger than 1; that’s fine. Only the **area** (integral) must equal 1.

---

## 2. Likelihood and why it becomes a product

### Probability model vs data distribution

In ML we often assume:

- There is some **true** (unknown) data distribution $p_\text{data}(x)$.
- We build a **model** distribution $p_\theta(x)$ with parameters.

We don’t know $p_{\text{data}}$ analytically; we only have samples:

$$
x^{(1)}, x^{(2)}, \dots, x^{(N)} \sim p_{\text{data}}.
$$

### Likelihood is a function of parameters

For one sample \(x\), the model assigns density $p_\theta(x)$.

If we treat \(x\) as fixed and \(\theta\) as variable, that quantity is called the **likelihood**.  
In other terms, **likelihood** measures how plausible a **parameter** \(\theta\) is, _given observed data_:

$$
\mathcal{L}(\theta; x) = p_\theta(x).
$$

For the whole dataset, we usually assume samples are i.i.d. (independent and identically distributed).  
That independence is exactly why the joint becomes a product:

$$
\mathcal{L}(\theta; x^{(1:N)})
= \prod_{i=1}^N p_\theta(x^{(i)}).
$$

**Why multiplication?**  
Because for independent events, joint probability (or joint density) factorizes:

$$
p(x^{(1)}, \ldots, x^{(N)}) = \prod_{i=1}^N p(x^{(i)}).
$$

### Log-likelihood turns products into sums

Products are numerically unstable and hard to optimize. Secondly, the likelihood is usually a small value and if we multiply all of them it might be near zero. So, We take logs:

$$
\log \mathcal{L}(\theta; x^{(1:N)})
= \sum_{i=1}^N \log p_\theta(x^{(i)}).
$$

So maximizing likelihood is the same as maximizing log-likelihood.

And minimizing **negative** log-likelihood (NLL) is the same objective with a “loss” sign.

---

## 3. Conditional probability, chain rule, and the Markov property

### Conditional probability

$$
p(a \mid b) = \frac{p(a,b)}{p(b)}.
$$

### Chain rule of probability

For variables \(x_1,\dots,x_T\):

$$
p(x_1,\dots,x_T)
= p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1,x_2)\cdots p(x_T\mid x_{1:T-1}).
$$

This is always true (no assumptions yet). It’s just repeated use of conditional probability.

### Markov property (the simplification we _choose_)

A **first-order Markov chain** assumes:

$$
p(x_t \mid x_{1:t-1}) = p(x_t \mid x_{t-1}).
$$

So the chain rule becomes:

$$
p(x_{1:T}) = p(x_1)\prod_{t=2}^T p(x_t \mid x_{t-1}).
$$

This is the _exact shape_ diffusion models use for the **forward noising process** and the **reverse denoising process**.

---

## 4. KL divergence (what it measures and where it comes from)

KL divergence compares two distributions. For discrete \(x\):

$$
D_{\mathrm{KL}}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}.
$$

For continuous \(x\):

$$
D_{\mathrm{KL}}(P\|Q) = \int P(x)\log\frac{P(x)}{Q(x)}\,dx.
$$

### The clean interpretation: “extra surprise”

If \(P\) is the truth and you pretend the world is \(Q\), KL measures how much extra “surprise” you expect (KL = cross-entropy − entropy).

Using the identity:

$$
D_{\mathrm{KL}}(P\|Q)
= \underbrace{\mathbb{E}_{x\sim P}[-\log Q(x)]}_{\text{cross-entropy}}
- \underbrace{\mathbb{E}_{x\sim P}[-\log P(x)]}_{\text{entropy}}.
$$

KL is the **gap** between cross-entropy and entropy.

### Why is KL always \(\ge 0\)?

This is a famous result (Gibbs’ inequality). Intuition:

- the “true” distribution is always best at describing samples from itself, on average.

So:

$$
D_{\mathrm{KL}}(P\|Q)\ge 0,\quad \text{and }=0 \text{ iff } P=Q \text{ (almost everywhere)}.
$$

### Why KL appears in maximum likelihood

Here’s the key bridge from “likelihood” to “KL”:

Start with the expected log-likelihood under the true data distribution:

$$
\mathbb{E}_{x\sim p_{\text{data}}}[\log p_\theta(x)].
$$

Now rewrite:

$$
D_{\mathrm{KL}}(p_{\text{data}}\|p_\theta)
= \mathbb{E}_{x\sim p_{\text{data}}}\left[\log \frac{p_{\text{data}}(x)}{p_\theta(x)}\right]
= \mathbb{E}_{x\sim p_{\text{data}}}[\log p_{\text{data}}(x)]
- \mathbb{E}_{x\sim p_{\text{data}}}[\log p_\theta(x)].
$$

The first term doesn’t depend on \(\theta\). So:

- minimizing $D_{\mathrm{KL}}(p_{\text{data}}\|p_\theta)$
- is equivalent to maximizing \(\mathbb{E}_{x\sim p_{\text{data}}}[\log p_\theta(x)]\).

That is **maximum likelihood**.

---

## 5. ELBO (the “evidence lower bound”) from scratch

People sometimes accidentally say “ELBOW”; the correct name is **ELBO**.

### The setup: latent-variable models

Assume a latent variable \(z\) (hidden) helps generate \(x\):

$$
p_\theta(x,z) = p_\theta(x\mid z)\,p(z).
$$

We want the marginal likelihood:

$$
p_\theta(x) = \int p_\theta(x,z)\,dz.
$$

That integral is often **intractable** meaning: there is no practical way to compute it exactly (either there is no closed-form solution, or doing it numerically would take an unrealistic amount of computation).

### What exactly is “intractable” here?

What’s intractable is the act of **integrating out the latent variables**.

- In a simple latent-variable model, \(z\) might be a single vector, and the integral can already be hard if \(z\) is high-dimensional.
- In **diffusion models**, the situation is more extreme because the “latent variable” is not one thing — it’s an entire **sequence** of latent states:
  $$
  x_1, x_2, \dots, x_T.
  $$

So the marginal likelihood of the data \(x_0\) looks like:

$$
p_\theta(x_0) = \int \int..... \int p_\theta(x_{0:T})\,dx_1\,dx_2\cdots dx_T.
$$

That’s not one integral it’s effectively **\(T\) integrals** (one for each timestep).

### Why does diffusion make it especially hard?

Each \(x_t\) has the same shape as the image.

So if your images are \(28\times 28\), then each $x_t$ lives in a **784-dimensional** space (more if you have channels like RGB).  
Now imagine integrating over a 784D variable… and then doing that again for $x_{t-1}$… and again… all the way up to \(x_T\).

If \(T=1000\) (a common choice), the full marginalization involves integrating over roughly:

- \(T\) latent variables,
- each in about \(784\) dimensions,

which is like integrating over a space with about \(784{,}000\) degrees of freedom.

That’s what we mean by **intractable**: the exact computation explodes in dimensionality, so we need an alternative (like **ELBO / variational bounds**) that avoids doing these massive integrals directly.

### Introduce an approximation $q_\phi(z\mid x)$

We _invent_ a distribution $q_\phi(z\mid x)$ to approximate the true posterior $p_\theta(z\mid x)$.

Now do the classic Jensen trick:

$$
\log p_\theta(x)
= \log \int p_\theta(x,z)\,dz
= \log \int q_\phi(z\mid x)\,\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz
= \log \mathbb{E}_{z\sim q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right].
$$

Apply Jensen’s inequality (\(\log \mathbb{E}[\cdot] \ge \mathbb{E}[\log \cdot]\)):

$$
\log p_\theta(x) \ge
\mathbb{E}_{z\sim q_\phi(z\mid x)}\left[\log p_\theta(x,z) - \log q_\phi(z\mid x)\right].
$$

That right-hand side is the **ELBO**.

### The “ELBO = reconstruction − KL” form

Expand \(p*\theta(x,z)=p*\theta(x\mid z)p(z)\):

$$
\mathrm{ELBO}(x) =
\mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
- D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z)).
$$

- First term: make \(z\) explain \(x\) well (reconstruction / data fit)
- Second term: keep \(q\) close to the prior \(p(z)\) (regularization)

### The “ELBO gap” identity

A very important identity:

$$
\log p_\theta(x) = \mathrm{ELBO}(x) + D_{\mathrm{KL}}(q_\phi(z\mid x)\|p_\theta(z\mid x)).
$$

Because KL is nonnegative, ELBO is a lower bound on \(\log p\_\theta(x)\).

---

## 6. Diffusion models as a latent-variable model

A diffusion model is a latent-variable model where the latent variables are a _whole sequence_:

$$
x_0, x_1, x_2, \dots, x_T.
$$

- \(x_0\): real data (image)
- \(x_T\): (almost) pure noise
- The intermediate \(x*1,\dots,x*{T-1}\): progressively noisier versions

### Forward process \(q\): fixed noising (a Markov chain)

We define a Markov chain that adds a little Gaussian noise at each step:

$$
q(x_{1:T}\mid x_0) = \prod_{t=1}^T q(x_t\mid x_{t-1}),
$$

with

$$
q(x_t\mid x_{t-1}) = \mathcal{N}\left(\sqrt{1-\beta_t}\,x_{t-1},\; \beta_t I\right),
$$

where \(0 < \beta_t < 1\) is a variance schedule.

Define \(\alpha*t = 1-\beta_t\), and \(\bar{\alpha}\_t = \prod*{s=1}^t \alpha_s\).

A crucial result: you can sample \(x_t\) directly from \(x_0\):

$$
q(x_t\mid x_0)=\mathcal{N}\left(\sqrt{\bar{\alpha}_t}\,x_0,\; (1-\bar{\alpha}_t)I\right),
$$

so equivalently:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,I).
$$

**Interpretation:**  
\(\varepsilon\) has the **same shape** as \(x_0\).  
For images, it’s “one Gaussian noise value per pixel (and channel)”.

### Reverse process \(p\_\theta\): learned denoising

We want to reverse the noising:

$$
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t).
$$

We choose a simple prior:

$$
p(x_T)=\mathcal{N}(0,I),
$$

and pick the forward schedule so that \(x_T\) is very close to that Gaussian.

Each reverse step is modeled as a Gaussian:

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}(\mu_\theta(x_t,t), \Sigma_t).
$$

The original DDPM often fixes \(\Sigma_t\) (or uses a simple parameterization), though later work also learns it.

### Where ELBO comes in (diffusion “VLB”)

We want to maximize \(\log p\_\theta(x_0)\), but that means integrating out all latent steps:

$$
\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})\,dx_{1:T},
$$

which is intractable directly.

So we do the same ELBO trick:

- choose a variational distribution over latents.
- in diffusion, we naturally pick the forward process \(q(x\_{1:T}\mid x_0)\).

This yields a variational lower bound (VLB) that decomposes into **a sum of KL terms across time steps**, comparing:

- the _true_ reverse posterior \(q(x\_{t-1}\mid x_t, x_0)\),
- versus the model \(p*\theta(x*{t-1}\mid x_t)\).

---

## 7. From diffusion ELBO to the simple MSE loss

Here’s the key simplification that made DDPMs practical:

Instead of directly predicting \(x\_{t-1}\), predict the noise \(\varepsilon\).

### The posterior is Gaussian (and tractable)

Because everything in the forward chain is Gaussian, the posterior

$$
q(x_{t-1}\mid x_t, x_0)
$$

is also Gaussian with a closed-form mean and variance (derived by Gaussian conditioning).

### Predicting noise is equivalent to predicting the mean

From

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon,
$$

if a network can estimate \(\varepsilon\) from \((x_t,t)\), it can estimate \(x_0\) and therefore the reverse mean.

DDPM uses a parameterization like:

$$
\mu_\theta(x_t,t)
= \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t,t)\right).
$$

### The “simple loss”

After algebra, the diffusion ELBO becomes (up to weights) an expectation of squared error between true noise and predicted noise:

$$
\mathcal{L}_{\text{simple}}
=
\mathbb{E}_{t\sim \mathrm{Uniform}\{1,\dots,T\},\,x_0\sim q(x_0),\,\varepsilon\sim \mathcal{N}(0,I)}
\left[
\left\|\varepsilon - \varepsilon_\theta(x_t,t)\right\|^2
\right],
$$

where \(x_t = \sqrt{\bar{\alpha}\_t}x_0 + \sqrt{1-\bar{\alpha}\_t}\varepsilon\).

**This is the punchline:**  
Diffusion training can be “just MSE” — but it’s MSE that comes from an ELBO on a Markov latent-variable model.

---

## 8. Sampling: the reverse process

Once trained, generation looks like this:

1. Sample \(x_T \sim \mathcal{N}(0,I)\).
2. For \(t=T,T-1,\dots,1\):
   - predict \(\varepsilon\_\theta(x_t,t)\),
   - compute \(\mu\_\theta(x_t,t)\),
   - sample \(x*{t-1}\sim \mathcal{N}(\mu*\theta(x_t,t), \Sigma_t)\)  
     (often with no extra noise at the final step).

---

## 9. Next: a 2D toy diffusion model you can _see_

Your idea is perfect for intuition: train diffusion on **2D points** so you can literally visualize the forward and reverse trajectories.

### Dataset (2D)

Pick a simple 2D distribution, e.g.

- two moons,
- a circle,
- a mixture of Gaussians.

So \(x_0 \in \mathbb{R}^2\).

### Forward process (same math)

Use the same:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon,\quad \varepsilon\sim \mathcal{N}(0,I),
$$

but now \(I\) is \(2\times 2\), and \(\varepsilon\) is a 2D Gaussian vector.

### Model

Use a small MLP:

$$
\varepsilon_\theta(x_t,t) \in \mathbb{R}^2.
$$

### Training loss

Same simple loss:

$$
\mathbb{E}\left[\|\varepsilon-\varepsilon_\theta(x_t,t)\|^2\right].
$$
