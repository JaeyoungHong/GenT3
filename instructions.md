# MM-Conv: A Spectral Framework for Coupled Channel-Mixing and 3×3 Convolution

## 1. Overview

MM-Conv is a parameterized linear operator that performs **matrix multiplication across the hidden dimension** and **3×3 convolution across the spatial dimensions** simultaneously. It generalizes the state-kernel construction of ConvT3 (Anonymous, ICLR 2026) by decoupling the universal spatial-diagonalization machinery (DST) from the per-channel-pair structure (TT ratios + spectral coefficients).

Every linear layer in any architecture (Q/K/V/O projections in attention, MLP up/down projections, the B/C/D kernels in an SSM) can be replaced by an MM-Conv block. The result is a network in which every linear layer simultaneously mixes channels and applies a 3×3 spatial convolution with TT structure, while the temporal architecture (attention, recurrence, etc.) is left completely free.

## 2. Mathematical Background

### 2.1 Tridiagonal Toeplitz (TT) matrices

A TT matrix `T = tridiag(l, d, u)` of size `N × N` has closed-form eigendecomposition. The `i`-th eigenvalue and the `j`-th entry of the `i`-th eigenvector are:

- `lambda_i = d + 2 * sqrt(l * u) * cos(i * pi / (N + 1))`
- `x[i, j] = (l / u)^(j / 2) * sin(i * j * pi / (N + 1))`

The eigenvector matrix `V` (with `V[j, i] = x[i, j]`) factors exactly as

    V = rho . Phi

where

- `Phi[j, i] = sin(i * j * pi / (N + 1))` is the **unnormalized Discrete Sine Transform** of order `N`, depending only on `N`
- `rho[j, j] = (l / u)^(j / 2)` is **diagonal**, depending only on the off-diagonal ratio `r = l / u`

Therefore every TT matrix of size `N` admits the decomposition

    T = rho . Phi . Lambda . Phi^{-1} . rho^{-1}

Useful identity: `Phi` is symmetric and satisfies `Phi^2 = ((N + 1) / 2) * I`, so `Phi^{-1} = (2 / (N + 1)) * Phi`. The DST and its inverse are the same operation up to scaling.

### 2.2 2D extension

For 2D inputs of size `H × W`, a 3×3 convolution operator with proportionality constraints (each 1D slice TT, with ratios `alpha_H` and `alpha_W` for the two axes) factors using the Kronecker structure:

    T_2D = (rho_H ⊗ rho_W) . (Phi_H ⊗ Phi_W) . Lambda_2D . (Phi_H ⊗ Phi_W)^{-1} . (rho_H ⊗ rho_W)^{-1}

where `Phi_H, Phi_W` are the two DSTs and `rho_H, rho_W` the two diagonal twists.

### 2.3 Key decoupling

The Phi matrices depend only on `(H, W)` — they have no learnable degrees of freedom. The `rho` twists depend only on the ratios `(alpha_H, alpha_W)`. The eigenvalues `Lambda_2D` carry the rest of the spectral content. So the spatial diagonalization machinery is split into:

- A **universal** part (the DSTs) — fixed, shared across the entire model
- A **parametric** part (`rho` and `Lambda`) — free to vary with the hidden-dimension structure

This makes the hidden dimension fully independent of the spatial diagonalization.

## 3. The MM-Conv Framework

### 3.1 Universal components (shared, fixed)

```
Phi_H[i_h, k_h] = sin(i_h * k_h * pi / (H + 1))      # shape: (H, H)
Phi_W[i_w, k_w] = sin(i_w * k_w * pi / (W + 1))      # shape: (W, W)
Psi_H[k_h, j_h] = (2 / (H + 1)) * Phi_H[k_h, j_h]    # shape: (H, H), inverse DST H
Psi_W[k_w, j_w] = (2 / (W + 1)) * Phi_W[k_w, j_w]    # shape: (W, W), inverse DST W
xi_H[k_h]       = cos(k_h * pi / (H + 1))            # shape: (H,)
xi_W[k_w]       = cos(k_w * pi / (W + 1))            # shape: (W,)
```

`i_h, j_h, k_h` index `{1, ..., H}` (1-indexed in the math, 0..H-1 in code as long as `+1` shifts are kept consistent). Same for the `W` indices.

These do not depend on `D_o` or `D_i` and can be precomputed once for each spatial resolution used in the model.

### 3.2 Hidden-dimension parameters (learnable, per block)

A single MM-Conv block carries **six** learnable matrices, all of shape `(D_o, D_i)`:

```
alpha_H[q, r] in C    # off-diagonal ratio for H axis
alpha_W[q, r] in C    # off-diagonal ratio for W axis
a[q, r]       in C    # spectral DC coefficient
b[q, r]       in C    # spectral H coefficient
c[q, r]       in C    # spectral W coefficient
d[q, r]       in C    # spectral H*W cross coefficient
```

These are the only block-level free parameters. The DST machinery is shared globally; the ratios and spectral coefficients live entirely on `(q, r)` and form genuine matrices in the hidden dimension.

### 3.3 Derived quantities

The forward and inverse spatial twists are derived from the ratios:

```
rho_H[q, r, j_h]   = alpha_H[q, r] ^ (j_h / 2)         # shape: (D_o, D_i, H)
rho_W[q, r, j_w]   = alpha_W[q, r] ^ (j_w / 2)         # shape: (D_o, D_i, W)
sigma_H[q, r, j_h] = alpha_H[q, r] ^ (-j_h / 2)        # shape: (D_o, D_i, H)
sigma_W[q, r, j_w] = alpha_W[q, r] ^ (-j_w / 2)        # shape: (D_o, D_i, W)
```

The spectral gain (per channel pair, per spectral location):

```
Lambda[q, r, k_h, k_w] = a[q, r]
                       + b[q, r] * xi_H[k_h]
                       + c[q, r] * xi_W[k_w]
                       + d[q, r] * xi_H[k_h] * xi_W[k_w]
```

Shape: `(D_o, D_i, H, W)`.

## 4. The MM-Conv Operator

### 4.1 Definition

Given an input tensor `X` of shape `(H, W, D_i)`, the MM-Conv operator produces an output `Y` of shape `(H, W, D_o)`:

```
Y[i_h, i_w, q] = sum over r, k_h, k_w, j_h, j_w of:
    rho_H[q, r, i_h] * rho_W[q, r, i_w]        # post-twist (matrix in (q,r), elementwise in i_h, i_w)
  * Phi_H[i_h, k_h] * Phi_W[i_w, k_w]          # forward DSTs (universal)
  * Lambda[q, r, k_h, k_w]                     # spectral gain (matrix in (q,r), elementwise in (k_h, k_w))
  * Psi_H[k_h, j_h] * Psi_W[k_w, j_w]          # inverse DSTs (universal)
  * sigma_H[q, r, j_h] * sigma_W[q, r, j_w]    # pre-twist (matrix in (q,r), elementwise in j_h, j_w)
  * X[j_h, j_w, r]                             # input
```

Free indices: `(i_h, i_w, q)`. Contracted indices: `(r, k_h, k_w, j_h, j_w)`.

### 4.2 Factored form

The `(q, r)` index pair stays threaded through every intermediate tensor. The `r` contraction happens only at the very end, together with the unlift. Implementing the operator in three stages:

**Stage 1 — Lift (per channel pair):**

```
Z[q, r, k_h, k_w] = sum over j_h, j_w of:
      Psi_H[k_h, j_h] * Psi_W[k_w, j_w]
    * sigma_H[q, r, j_h] * sigma_W[q, r, j_w]
    * X[j_h, j_w, r]
```

Shape: `(D_o, D_i, H, W)`. No contraction over `r` yet.

**Stage 2 — Spectral gain (elementwise):**

```
Omega[q, r, k_h, k_w] = Lambda[q, r, k_h, k_w] * Z[q, r, k_h, k_w]
```

Shape: `(D_o, D_i, H, W)`. Pure Hadamard product.

**Stage 3 — Unlift and contract `r`:**

```
Y[i_h, i_w, q] = sum over r, k_h, k_w of:
      rho_H[q, r, i_h] * rho_W[q, r, i_w]
    * Phi_H[i_h, k_h] * Phi_W[i_w, k_w]
    * Omega[q, r, k_h, k_w]
```

Shape: `(H, W, D_o)`.

### 4.3 Equivalence to a 3×3 convolution

For each fixed `(q, r)`, the operator's `(q, r)` slice is a genuine 3×3 convolution kernel with TT structure. The 9 entries of the kernel `K[q, r, dh, dw]` (with `dh, dw in {-1, 0, 1}`) are determined by `(a[q, r], b[q, r], c[q, r], d[q, r], alpha_H[q, r], alpha_W[q, r])`:

```
K[q, r,  0,  0] = a[q, r]
K[q, r,  0,  1] = b[q, r] / 2
K[q, r,  0, -1] = alpha_W[q, r] * b[q, r] / 2
K[q, r,  1,  0] = c[q, r] / 2
K[q, r, -1,  0] = alpha_H[q, r] * c[q, r] / 2
K[q, r,  1,  1] = d[q, r] / 4
K[q, r,  1, -1] = alpha_W[q, r] * d[q, r] / 4
K[q, r, -1,  1] = alpha_H[q, r] * d[q, r] / 4
K[q, r, -1, -1] = alpha_H[q, r] * alpha_W[q, r] * d[q, r] / 4
```

(The factors of 1/2 and 1/4 come from the cosine-to-side mapping in the TT eigendecomposition; verify against the implementation's specific DST normalization.)

Thus an MM-Conv block is mathematically equivalent to a standard 3×3 convolution layer with `D_o × D_i` channels, parameterized via 6 scalars per channel pair instead of the unconstrained 9. The structured parameterization buys spectral analysis, exact spatial diagonalization, and explicit ratio control.

## 5. Special Cases

### 5.1 Pure matrix multiplication (1×1 convolution)

Set `b = c = d = 0`, `a = W` (the desired channel-mixing matrix). The ratios `alpha_H, alpha_W` become irrelevant (their contributions cancel since the spectral gain has no `xi` terms). The operator collapses to:

```
Y[i_h, i_w, q] = W[q, r] * X[i_h, i_w, r]
```

This is the cheapest setting and corresponds to a pointwise linear layer.

### 5.2 Depthwise 3×3 convolution

Set `D_o = D_i = P` and make all six matrices diagonal in `(q, r)` (i.e. nonzero only when `q == r`). Each channel evolves under its own 3×3 spatial kernel; no channel mixing.

### 5.3 Full 3×3 convolution with channel mixing

All six matrices `(D_o × D_i)` are dense. This is the most expressive setting.

## 6. Building Architectures with MM-Conv

The MM-Conv block is a drop-in replacement for any "linear layer" in a deep network. The spatial 3×3 conv structure and the channel mixing happen in one fused operator.

### 6.1 Multi-head causal time attention

Let `X` have shape `(L, H, W, P)` where `L` is the sequence length. Use MM-Conv blocks for the Q, K, V, O projections. Each block has its own six hidden-dim matrices.

**Projections.** For each time step `t`, apply MM-Conv to `X[t]`:

```
Q[t, i_h, i_w, h, e] = MMConv_Q[i_h, i_w; j_h, j_w; (h, e), p] (X[t, j_h, j_w, p])
K[t, i_h, i_w, h, e] = MMConv_K[i_h, i_w; j_h, j_w; (h, e), p] (X[t, j_h, j_w, p])
V[t, i_h, i_w, h, e] = MMConv_V[i_h, i_w; j_h, j_w; (h, e), p] (X[t, j_h, j_w, p])
```

Here `(h, e)` flattens to `D_o = H_heads * d_head` and `p` ranges over `D_i = P`.

**Attention (per spatial location, along time).** For each `(i_h, i_w, h)`, compute scores between time steps:

```
score[t, s, i_h, i_w, h] = sum_e (Q[t, i_h, i_w, h, e] * K[s, i_h, i_w, h, e]) / sqrt(d_head)
beta[t, s, i_h, i_w, h]  = softmax_over_s_with_causal_mask(score[t, :, i_h, i_w, h])[s]
```

**Aggregation and output projection.**

```
A[t, i_h, i_w, h, e] = sum_{s <= t} beta[t, s, i_h, i_w, h] * V[s, i_h, i_w, h, e]
Y[t, i_h, i_w, q]    = MMConv_O[i_h, i_w; j_h, j_w; q, (h, e)] (A[t, j_h, j_w, h, e])
```

Each MMConv_{Q,K,V,O} carries its own six hidden-dim matrices. The spatial mixing happens implicitly in every projection; attention itself only mixes along time.

Alternative attention designs (spatially mixed attention, full spatiotemporal attention, etc.) are straightforward variants — just change which indices the softmax sums over.

### 6.2 Position-wise MLP

```
H[t, i_h, i_w, m] = activation( MMConv_1[i_h, i_w; j_h, j_w; m, p] (X[t, j_h, j_w, p]) )
Y[t, i_h, i_w, q] = MMConv_2[i_h, i_w; j_h, j_w; q, m] (H[t, j_h, j_w, m])
```

`m` is the MLP hidden dimension. Two MM-Conv blocks with a nonlinearity between.

### 6.3 Transformer block

A standard transformer block becomes:

```
def transformer_block(X):
    # Attention sublayer
    X = X + Attention(LayerNorm(X))   # uses MM-Conv for Q/K/V/O
    # MLP sublayer
    X = X + MLP(LayerNorm(X))         # uses MM-Conv for both projections
    return X
```

Stack as many of these as needed. The spatial 3×3 conv structure is built into every linear layer; the temporal architecture is whatever you want.

### 6.4 As an SSM state kernel

The framework also accommodates the original ConvT3-style use as a state kernel `A` in a ConvSSM:

```
X[t+1, :, :, :] = MMConv_A(X[t, :, :, :]) + MMConv_B(U[t, :, :, :])
Y[t, :, :, :]   = MMConv_C(X[t, :, :, :]) + MMConv_D(U[t, :, :, :])
```

Parallel scans work along the time axis since each `(q, r)` slice of `A` is a structured operator with a known spectral form. For the special case where `A` is channel-diagonal (`D_o = D_i = P` with `a, b, c, d, alpha_H, alpha_W` all diagonal in `(q, r)`), and the eigenvalues `Lambda[p, p, :, :]` are made Hurwitz via softmax-based stability constraints (see ConvT3 paper Section 3.3), this recovers ConvT3 exactly.

## 7. Implementation Reference

### 7.1 Tensor shapes summary

| Symbol            | Shape                    | Notes                                  |
|-------------------|--------------------------|----------------------------------------|
| `X`               | `(H, W, D_i)`            | input                                  |
| `Y`               | `(H, W, D_o)`            | output                                 |
| `Phi_H, Psi_H`    | `(H, H)`                 | precomputed, real                      |
| `Phi_W, Psi_W`    | `(W, W)`                 | precomputed, real                      |
| `xi_H`            | `(H,)`                   | precomputed, real                      |
| `xi_W`            | `(W,)`                   | precomputed, real                      |
| `alpha_H, alpha_W`| `(D_o, D_i)`             | learnable, complex                     |
| `a, b, c, d`      | `(D_o, D_i)`             | learnable, complex                     |
| `rho_H, sigma_H`  | `(D_o, D_i, H)`          | derived from `alpha_H`                 |
| `rho_W, sigma_W`  | `(D_o, D_i, W)`          | derived from `alpha_W`                 |
| `Lambda`          | `(D_o, D_i, H, W)`       | derived from `a, b, c, d, xi_H, xi_W`  |
| `Z, Omega`        | `(D_o, D_i, H, W)`       | intermediates                          |

For batched data with batch dim `B` (and possibly time `L`), add those as leading dimensions on `X, Y, Z, Omega` (e.g. `X` is `(B, L, H, W, D_i)`).

### 7.2 Forward pass (pseudocode)

```python
def mmconv_forward(X, alpha_H, alpha_W, a, b, c, d,
                   Phi_H, Phi_W, Psi_H, Psi_W, xi_H, xi_W):
    """
    X:        (..., H, W, D_i)
    Phi_H:    (H, H)   forward DST for H
    Phi_W:    (W, W)   forward DST for W
    Psi_H:    (H, H)   inverse DST for H
    Psi_W:    (W, W)   inverse DST for W
    xi_H:     (H,)
    xi_W:     (W,)
    alpha_H:  (D_o, D_i)
    alpha_W:  (D_o, D_i)
    a, b, c, d: each (D_o, D_i)

    Returns:
    Y:        (..., H, W, D_o)
    """
    # ---- Derive twists ----
    # j_h ranges 1..H, j_w ranges 1..W. Use float indices.
    jH = arange(1, H + 1).float()                          # (H,)
    jW = arange(1, W + 1).float()                          # (W,)

    # rho/sigma: (D_o, D_i, H) and (D_o, D_i, W)
    rho_H   = alpha_H[..., None] ** (jH / 2)               # (D_o, D_i, H)
    rho_W   = alpha_W[..., None] ** (jW / 2)               # (D_o, D_i, W)
    sigma_H = alpha_H[..., None] ** (-jH / 2)              # (D_o, D_i, H)
    sigma_W = alpha_W[..., None] ** (-jW / 2)              # (D_o, D_i, W)

    # ---- Spectral gain ----
    # Lambda[q, r, k_h, k_w] = a + b*xi_H[k_h] + c*xi_W[k_w] + d*xi_H[k_h]*xi_W[k_w]
    Lambda = (a[..., None, None]
            + b[..., None, None] * xi_H[None, None, :, None]
            + c[..., None, None] * xi_W[None, None, None, :]
            + d[..., None, None] * (xi_H[None, None, :, None] * xi_W[None, None, None, :]))
    # Lambda: (D_o, D_i, H, W)

    # ---- Stage 1: lift ----
    # X':  (..., H, W, D_o, D_i) = X * sigma_H * sigma_W, broadcast in (q, r)
    # Naively this materializes a (..., H, W, D_o, D_i) tensor.
    # Optimisations exist; the cleanest reference form follows.

    # Broadcast X to (..., 1, 1, H, W, D_i) then multiply by sigma over (D_o, D_i, j_h, j_w)
    Xp = X[..., None, None, :, :, :]                       # (..., 1, 1, H, W, D_i)
    sH = sigma_H.transpose()                               # (H, D_o, D_i) conceptually
    # Apply sigma_H, sigma_W to spatial dims of X, indexed by (q, r):
    # Xt[..., q, r, j_h, j_w, r'] but we want r' == r, so:
    # We'll do it via einsum-style operations.

    # Cleaner: build the twisted input
    #   Xtwisted[..., q, r, j_h, j_w] = sigma_H[q, r, j_h] * sigma_W[q, r, j_w] * X[..., j_h, j_w, r]
    Xtwisted = einsum('qrh,qrw,...hwr->...qrhw', sigma_H, sigma_W, X)
    # Xtwisted: (..., D_o, D_i, H, W)

    # Apply inverse DSTs:
    # Z[..., q, r, k_h, k_w] = sum_{j_h, j_w} Psi_H[k_h, j_h] * Psi_W[k_w, j_w] * Xtwisted[..., q, r, j_h, j_w]
    Z = einsum('kh,lw,...qrhw->...qrkl', Psi_H, Psi_W, Xtwisted)
    # Z: (..., D_o, D_i, H, W)  (k_h, k_w renamed (k, l) in einsum)

    # ---- Stage 2: spectral gain ----
    Omega = Lambda * Z                                     # (..., D_o, D_i, H, W)

    # ---- Stage 3: unlift + contract r ----
    # Apply forward DSTs:
    # Omega_spatial[..., q, r, i_h, i_w] = sum_{k_h, k_w} Phi_H[i_h, k_h] * Phi_W[i_w, k_w] * Omega[..., q, r, k_h, k_w]
    Omega_spatial = einsum('ik,jl,...qrkl->...qrij', Phi_H, Phi_W, Omega)
    # Omega_spatial: (..., D_o, D_i, H, W)

    # Multiply by rho_H, rho_W and contract over r:
    # Y[..., i_h, i_w, q] = sum_{r} rho_H[q, r, i_h] * rho_W[q, r, i_w] * Omega_spatial[..., q, r, i_h, i_w]
    Y = einsum('qrh,qrw,...qrhw->...hwq', rho_H, rho_W, Omega_spatial)

    return Y
```

Notes on the pseudocode:

- The reference implementation materializes a `(D_o, D_i, H, W)` intermediate tensor (`Z`, `Omega`), which is memory-intensive. For large channel counts, prefer one of the optimization paths in 7.4.
- The `einsum` calls use single-character indices for clarity: `h, w` for spatial input, `k, l` for spectral, `i, j` for spatial output, `q, r` for hidden, `...` for leading batch/time dims.
- All learnable params (`alpha_H, alpha_W, a, b, c, d`) are complex. See 7.5 for the real-valued parameterization that ensures real outputs.

### 7.3 Backward pass

PyTorch / JAX autograd handles this automatically through the einsum and elementwise operations. No custom backward is needed for the reference implementation.

### 7.4 Optimisations

The reference implementation has complexity `O(D_o * D_i * H * W * max(H, W))` per call (dominated by the einsums that mix spatial axes with `(q, r)`). Three optimisation routes:

**(a) Materialize a standard 3×3 kernel.** For each `(q, r)`, compute the 9 kernel entries from `(a, b, c, d, alpha_H, alpha_W)` using the formulas in Section 4.3, then call standard 3×3 convolution. Cost: `O(9 * D_o * D_i * H * W)`. Loses spectral interpretation at compute time but is the fastest practical path on GPUs with optimized conv kernels.

**(b) Decomposed spectral form.** `Lambda` has rank ≤ 4 structure in `(k_h, k_w)`. Decompose:

    Lambda = a (x) 1_H (x) 1_W + b (x) xi_H (x) 1_W + c (x) 1_H (x) xi_W + d (x) xi_H (x) xi_W

This lets you compute four cheaper spectral-domain products and sum them, but it still needs per-`(q, r)` DSTs unless you separately tackle (c).

**(c) Approximate / share `alpha`.** If the `alpha_H, alpha_W` matrices are constrained to be uniform across `(q, r)` (single scalar per axis, like the original ConvT3 setting), the lift and unlift become channel-independent and can be applied once via a fast DST in `O(D * H * W * log H)`. This recovers the original ConvT3 cost, at the cost of flexibility.

For most practical configurations, route (a) is recommended.

### 7.5 Real-valued parameterization

To keep `X` and `Y` real-valued with complex parameters, follow the conjugate-pair convention used in S5 / ConvS5 / ConvT3:

- Parameterize each complex matrix `M` as a pair `(M_re, M_im)` of real matrices, both shape `(D_o, D_i)`.
- Pair the hidden dimension `D_i` (resp. `D_o`) into conjugate pairs, ensuring complex outputs aggregate to real numbers.
- Equivalently, parameterize all six matrices as real and take `alpha_H, alpha_W` as real scalars per `(q, r)`. For the special case `alpha_H, alpha_W = ±1` (the regime used in the ConvT3 experiments), all twists become real and `|rho| = 1`, so no conditioning issues arise.

### 7.6 Stability (when used as an SSM state kernel)

If MM-Conv is used as the state operator `A` in a recurrent SSM (rather than as a linear layer in a feedforward / attention stack), the eigenvalues of `Lambda` must be Hurwitz (negative real part) for stable continuous-time dynamics, or `|.| < 1` for stable discrete-time dynamics.

The ConvT3 paper achieves this by:

1. Fixing `a = 1` and constraining `b, c, d` via softmax so that the bilinear form `phi(theta_H, theta_W) = a + b * cos(theta_H) + c * cos(theta_W) + d * cos(theta_H) * cos(theta_W)` is positive at the four corners `cos = ±1`, hence positive everywhere on the discrete grid.
2. Reparameterizing `Lambda_diagonal` with negative-real-part eigenvalues via `Re{Lambda} := -softplus(...)`.

For non-recurrent uses (attention, MLP), no stability constraint is needed.

### 7.7 Initialization

To recover behaviour close to a standard initialization:

- `a` ~ small random (or as a 1×1-conv-style initializer, e.g. Xavier/Kaiming over the `(D_o, D_i)` matrix).
- `b = c = d = 0` at init. This makes the block behave as a pure 1×1 conv (MM only) at the start of training.
- `alpha_H, alpha_W = -1` (skew-symmetric spatial coupling, the setting used in ConvT3 experiments).

With these settings, an MM-Conv block at init is indistinguishable from a 1×1 linear layer, and learns spatial mixing gradually as `b, c, d` move away from zero.

## 8. Recommended Implementation Plan

For a first reference implementation:

1. **Module class.** Implement `MMConv(d_in, d_out, H, W, alpha_init=-1.0)` as an `nn.Module` (PyTorch) or equivalent. Hold the six learnable matrices as parameters. Precompute and buffer `Phi_H, Phi_W, Psi_H, Psi_W, xi_H, xi_W` based on `(H, W)`.
2. **Forward pass.** Implement the three-stage factored form (Section 4.2) using `einsum`. Verify against the single-equation form (Section 4.1) on small tensors.
3. **Unit tests.**
   - Equivalence to 1×1 conv when `b = c = d = 0`: random `a`, verify `Y[i_h, i_w, q] == a[q, r] * X[i_h, i_w, r]`.
   - Equivalence to a standard 3×3 conv when the kernel is materialised via Section 4.3 formulas and compared to a vanilla `Conv2d(d_in, d_out, 3, padding=1)`.
   - Linearity in `X`.
   - Gradient check via `torch.autograd.gradcheck`.
4. **Attention block.** Build `MMConvAttention(d_model, n_heads, H, W)` using four MM-Conv layers for Q/K/V/O and standard scaled-dot-product attention along time.
5. **MLP block.** Build `MMConvMLP(d_model, d_hidden, H, W)` using two MM-Conv layers with GELU between.
6. **Transformer block + stack.** Combine into a full transformer with residual + layer norm. Compare against a vanilla transformer with the same parameter budget on a small spatiotemporal task (e.g. Moving MNIST).
7. **Optimisations.** Once correctness is verified, switch the inner forward to the materialized-kernel path (Section 7.4 (a)) for speed.

## 9. Notation Cheatsheet

| Index | Range            | Meaning                            |
|-------|------------------|------------------------------------|
| `t`   | `1..L`           | time step                          |
| `i_h, j_h, k_h` | `1..H` | height (signal / signal / spectral) |
| `i_w, j_w, k_w` | `1..W` | width (signal / signal / spectral)  |
| `q`   | `1..D_o`         | output channel                     |
| `r`   | `1..D_i`         | input channel                      |
| `h`   | `1..H_heads`     | attention head                     |
| `e`   | `1..d_head`      | per-head dimension                 |
| `m`   | `1..d_mlp`       | MLP hidden                         |

`q` is always an output / row index, `r` always an input / column index. Together they form the matrix in the hidden dimension.

---

End of specification.