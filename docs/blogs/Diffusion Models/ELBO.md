# ELBO derivation (step-by-step)

**Blog by : Abhijit Challapalli**

This derivation follows the exact mathematical steps, with short explanations around each equation.  
We use:

- Observed variable: $x$
- Latent variable: $z$
- Approximate posterior (encoder): $q_\phi(z\mid x)$
- True posterior: $p_\theta(z\mid x)$
- Joint model: $p_\theta(z,x)=p_\theta(x\mid z)\,p_\theta(z)$

---

**Step 1: Start from KL divergence (always $\ge 0$)**

$$
D_{\mathrm{KL}}\!\left(q_\phi \,\|\, p_\theta\right)
=
\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\right]
$$

This KL measures how close $q_\phi(z\mid x)$ is to the true posterior $p_\theta(z\mid x)$.

---

**Step 2: Split the log fraction**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z\mid x)\big]
$$

Using $\log\frac{a}{b}=\log a-\log b$ and linearity of expectation.

---

**Step 3: Substitute Bayes’ rule for the true posterior**

Using

$$
p_\theta(z\mid x)=\frac{p_\theta(z,x)}{p_\theta(x)},
$$

we get:

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{p_\theta(z,x)}{p_\theta(x)}\right]
$$

---

**Step 4: Split the log again**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x)\big]
$$

---

**Step 5: Write the last expectation as an integral**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
+
\int q_\phi(z\mid x)\,\log p_\theta(x)\,dz
$$

This uses $\mathbb{E}_{q}[f(z)]=\int q(z)f(z)\,dz$.

---

**Step 6: Pull $\log p_\theta(x)$ outside the integral**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
+
\log p_\theta(x)\int q_\phi(z\mid x)\,dz
$$

Because $\log p_\theta(x)$ does not depend on $z$.

---

**Step 7: Use $\int q_\phi(z\mid x)\,dz = 1$**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
+
\log p_\theta(x)
$$

---

**Step 8: Rearrange to isolate $\log p_\theta(x)$**

$$
\log p_\theta(x)
=
-\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
+
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)
$$

This matches the “component 1 + component 2” view: the last term is a KL divergence (non-negative).

---

**Step 9: Drop the non-negative KL term to get a lower bound**

Since

$$
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)\ge 0,
$$

we have:

$$
\log p_\theta(x)
\ge
-\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
$$

---

**Step 10: Define the ELBO**

$$
\mathrm{ELBO}(x)
=
-\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z,x)\big]
$$

---

**Step 11: Expand the joint term $\log p_\theta(z,x)$**

Using $p_\theta(z,x)=p_\theta(x\mid z)p_\theta(z)$:

$$
\mathrm{ELBO}(x)
=
-\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x\mid z)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z)\big]
$$

---

**Step 12: Reorder terms (same as in the image)**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x\mid z)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log q_\phi(z\mid x)\big]
+
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(z)\big]
$$

---

**Step 13: Combine into a single log fraction**

$$
=
\mathbb{E}_{q_\phi(z\mid x)}\!\big[\log p_\theta(x\mid z)\big]
-
\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{q_\phi(z\mid x)}{p_\theta(z)}\right]
$$

---

**Final note: recognize the KL term and interpret the objective**

The last expectation is exactly a KL divergence:

$$
\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log \frac{q_\phi(z\mid x)}{p_\theta(z)}\right]
=
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right).
$$

So the ELBO can be written in the common VAE form:

$$
\mathrm{ELBO}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right).
$$

**Training:** we maximize $\mathrm{ELBO}(x)$, which is equivalent to minimizing the negative ELBO:

$$
-\mathrm{ELBO}(x)
=
-\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
+
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z)\right).
$$

- The first term $-\mathbb{E}_{q_\phi}[\log p_\theta(x\mid z)]$ is the **reconstruction negative log-likelihood (NLL)**, so we **minimize** it (equivalently maximize the reconstruction log-likelihood).
- The second term is a KL divergence to the prior, so the objective also pushes $q_\phi(z\mid x)$ to stay close to $p_\theta(z)$.

In short: **max ELBO** $\Leftrightarrow$ **min reconstruction NLL + min KL-to-prior** (a trade-off optimized jointly).
