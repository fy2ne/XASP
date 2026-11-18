# XASP Mathematical Formulation & Proofs

This document summarizes the two-layer spectral protocol implemented in `xasp-math` for 128-lane physical TPU tiles staged in Host DRAM by `google/tpu-sync`.

---

## 1. Layer I: Christoffel Ridge Leverage Stratification

Let $K, V \in \mathbb{R}^{N \times 128}$ be the layer-$\ell$ Key and Value slices aligned along TPU's 128-lane minor tile axis (`d_k = 128`). For any future Decode query $q \in \mathbb{R}^{128}$, evaluate each row's $\mu$-Ridge Leverage Score (the reciprocal of the regularized Christoffel function $\Lambda_\mu(k_i)$):

$$\tau_i(\mu) = k_i^\top \left(K^\top K + \mu I_{128}\right)^{-1} k_i = \frac{1}{\Lambda_\mu(k_i)} \in (0, 1), \qquad \mu = \frac{1}{N}\|K\|_F^2$$

### Proposition 1 (Query-Agnostic Uniform Logit Energy Bound)

For any query $q \in \mathbb{R}^{128} \setminus \{0\}$, the worst-case fraction of regularized logit energy captured by row $i \in \{1, \dots, N\}$ equals $\tau_i(\mu)$:

$$\sup_{q \in \mathbb{R}^{128} \setminus \{0\}} \frac{|\langle q, k_i \rangle|^2}{q^\top (K^\top K + \mu I_{128}) q} = k_i^\top \left(K^\top K + \mu I_{128}\right)^{-1} k_i = \tau_i(\mu)$$

**Proof.** Let $\Sigma_\mu = K^\top K + \mu I_{128}$. Applying Cauchy-Schwarz to $\langle \Sigma_\mu^{1/2} q, \Sigma_\mu^{-1/2} k_i \rangle$ yields $|\langle q, k_i \rangle|^2 \le (q^\top \Sigma_\mu q)(k_i^\top \Sigma_\mu^{-1} k_i) = (q^\top \Sigma_\mu q)\tau_i(\mu)$, with equality attained at $q \propto \Sigma_\mu^{-1} k_i$. $\blacksquare$

### Fast Sketch & Two-Stratum TCP Assignment

Using a subsampled $r = 32$ sketch $\tilde{K} \in \mathbb{R}^{N \times 32}$, `xasp-math` solves $\tilde{K}^\top \tilde{K} + \mu I_{32}$ in $O(N r^2 + r^3)$ to obtain approximate scores $\tilde{\tau}_i(\mu)$. Locking the initial 4 Attention Sinks ($i < 4$) and ranking rows by

$$s_i = \infty \cdot \mathbb{I}(i < 4) + \tilde{\tau}_i(\mu) \cdot \|v_i\|_2$$

splits the tile into:
- **Stratum-0 ($\mathcal{S}_0$, top $\alpha = 5\%$ rows by $s_i$):** Symmetric `INT8` quantized rows transmitted redundantly across Priority TCP Streams `0..1`.
- **Stratum-1 ($\mathcal{S}_1$, remaining $95\%$ rows):** Rotated and compressed via Layer II and striped across Bulk TCP Streams `2..15`.

---

## 2. Layer II: In-Place 128-Lane FWHT + Polar/QJL Over $K=16$ Striped TCP

Let $x \in \mathbb{R}^{128}$ be a Stratum-1 Key or Value row along the 128-lane physical minor tile axis.

### 2.1 Subsampled Randomized Hadamard Transform (SRHT)

Define the normalized $128 \times 128$ Walsh-Hadamard matrix $H_{128} = H_2^{\otimes 7} / \sqrt{128}$ and deterministic Rademacher sign diagonal $D = \text{diag}(\sigma_1, \dots, \sigma_{128})$ derived from `request_id_lo ^ layer_id`:

$$\Phi = H_{128} D \in \mathbb{R}^{128 \times 128}, \qquad \Phi^\top \Phi = \Phi \Phi^\top = I_{128}$$

Evaluating $\tilde{x} = \Phi x$ executes $128 \log_2 128 = 896$ additions and subtractions in-place along the 128-lane axis, preserves inner products ($\langle \Phi q, \Phi x \rangle = \langle q, x \rangle$), and disperses isolated channel spikes across all 128 coordinates.

### 2.2 Two-Stage Polar Lloyd-Max + 1-Bit QJL

- **Stage A:** $(b-1)$-bit symmetric Gaussian Lloyd-Max quantizer $Q_{b-1}(\tilde{x})$ scaled by $\rho_x / \sqrt{128}$, where $\rho_x = \|\tilde{x}\|_2$ (`f16`).
- **Stage B:** 1-bit QJL sign sketch $z_x = \text{sign}(\Psi r_x) \in \{-1, +1\}^{128}$ (`16 bytes`) computed on the residual $r_x = \tilde{x} - Q_{b-1}(\tilde{x})$ with $\rho_r = \|r_x\|_2$ (`f16`) and $\Psi = H_{128} D^{(2)}$.

### 2.3 Deadline Unbiased Estimator Under Striped TCP Erasures

The 128 rotated coordinates are partitioned into 14 disjoint subsets $\Omega_2, \dots, \Omega_{15}$ across Bulk TCP Streams `2..15`. At layer deadline $\tau_l$, let $\xi \in \{0, 1\}^{128}$ indicate coordinates whose stream arrived by $\tau_l$, $\Xi = \text{diag}(\xi)$, and missing coordinate fraction $\hat{\delta} = 1 - \frac{1}{128}\sum_{j=1}^{128} \xi_j$. Decode immediately reconstructs:

$$\hat{x}_{\delta} = \Phi^\top \left[ \frac{1}{1 - \hat{\delta}} \, \Xi \, \left( Q_{b-1}(\tilde{x}) + \sqrt{\frac{\pi}{2 \cdot 128}} \, \rho_r \, \Psi^\top z_x \right) \right]$$

### Theorem 1 (Exact Unbiasedness & Variance Bound Under Deadline Cutoffs)

Conditioned on stalled coordinate fraction $\hat{\delta} = \delta \in [0, 1)$ at $\tau_l$, $\widehat{\langle q, x \rangle}_\delta = \langle q, \hat{x}_\delta \rangle$ satisfies:

$$\mathbb{E}_{\Phi,\Psi,\xi}\!\left[\widehat{\langle q, x \rangle}_\delta\right] = \langle q, x \rangle, \qquad \text{Var}\!\left(\widehat{\langle q, x \rangle}_\delta\right) \le \frac{1}{128(1-\delta)}\left(\delta + \frac{\pi}{2}\gamma_{b-1}^2\right)\|q\|_2^2 \|x\|_2^2$$

where $\gamma_1 \approx 0.3634$ ($b=2$), $\gamma_2 \approx 0.1175$ ($b=3$), and $\gamma_3 \approx 0.0345$ ($b=4$).

**Proof.** Seeding $D, D^{(2)}$ per `(request_id, layer_id)` decouples coordinate placement from TCP socket congestion, giving $\mathbb{E}_\xi[\xi_j/(1-\delta)] = 1$. Combining the Gaussian sign covariance identity $\mathbb{E}[\text{sign}(g)h] = \sqrt{2/\pi}\,\text{Cov}(g,h)/\sigma$ with orthogonality $\Phi^\top\Phi = I_{128}$ cancels quantization residual bias in expectation. Summing QJL variance and coordinate subsampling variance over $(1-\delta)\cdot 128$ arrived coordinates gives the variance bound. $\blacksquare$

### Theorem 2 (Softmax Attention Distortion Bound)

Because Streams `0..1` deliver Stratum-0 anchors redundantly with zero loss ($\delta_0 = 0$) and Stratum-1 carries tail softmax mass $P_{\mathcal{S}_1} = \sum_{i \in \mathcal{S}_1} p_i$, Decode attention error obeys:

$$\mathbb{E}\!\left[ \|\text{Attn}(q, \hat{K}_\delta, \hat{V}_\delta) - \text{Attn}(q, K, V)\|_2 \right] \le P_{\mathcal{S}_1} \, \|V_{\mathcal{S}_1}\|_{2,\infty} \, \sqrt{\frac{\delta + \frac{\pi}{2}\gamma_{b-1}^2}{1-\delta}} \left( \frac{\|q\|_2 \|K_{\mathcal{S}_1}\|_{2,\infty}}{128} + \frac{1}{\sqrt{128}} \right)$$

**Proof Sketch.** Perturbation is restricted to $\mathcal{S}_1$ ($\delta_0 = 0$). Applying softmax Lipschitz stability $\|(\text{diag}(p) - pp^\top)\eta\|_1 \le 2 P_{\mathcal{S}_1}\max_{i \in \mathcal{S}_1}|\eta_i|$ with Theorem 1 bounds relative inner-product RMSE below $2.80\%$ when 1 of 14 bulk sockets misses $\tau_l$ at $b=3$ bits. $\blacksquare$
