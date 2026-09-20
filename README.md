# Second-Order Differential Uniformity of Boolean Functions

Code accompanying a thesis on the second-order differential uniformity of vectorial Boolean functions over `GF(2^n)`. The scripts search for, classifies and analyses functions that achieve the theoretical optimum for this property.

## Background: second-order differential uniformity

For a function `F : GF(2)^n -> GF(2)^n`, its **second-order derivative** with respect to two vectors `a, b` is

```
D_a D_b F(x) = F(x) ⊕ F(x⊕a) ⊕ F(x⊕b) ⊕ F(x⊕a⊕b)
```

For each linearly independent pair `(a, b)` (equivalently, each 2-dimensional subspace `{0, a, b, a⊕b}`), this derivative takes each output value `c` some number of times as `x` ranges over the domain. The **second-order differential uniformity**, `δ²(F)`, is the largest such count, taken over every choice of `a`, `b`, and `c`.

`δ²(F)` measures how far `F` is from behaving like an evenly distributed function under second-order differentials. A low value means the function is more resistant to differential attacks, which is desirable for S-boxes in symmetric-key encryption algorithms. For functions on an even number of input bits, the smallest attainable value is `δ² = 4`. A function reaching this bound is called **optimal**. The programs in this repository search for such optimal functions and studies their structure (e.g. multiplicity spectra).

## Programs

All scripts read their parameters from [config.ini](config.ini) and are run directly, e.g. `python search.py`. Results are saved to CSV files (or, for `grassmann.py`, a text file) in the working directory.

- **[functions.py](functions.py)** — Shared functions used by other scripts: `GF(2^n)` exponential/logarithm tables from primitive polynomials, truth-table and ANF computation (via the butterfly/Möbius transform), GF(2) linear algebra (Gaussian elimination for rank, inverse and kernel basis), canonical-form reduction of a cubic ANF's coefficients, enumeration of 2-dimensional subspaces (`get_AB`, `get_AB_subfield`), and construction of the coefficient matrix `M` whose kernel encodes a function's second-order derivative on a given subspace.

- **[search.py](search.py)** — Exhaustive search over small fields (`n` = 3–5). Checks every cubic Boolean function up to affine equivalence (in canonical form) and tests each one's second-order differential uniformity directly against every 2D subspace, recording the optimal (`δ² = 4`) functions to `optimal.csv`.

- **[classify.py](classify.py)** — Takes the optimal functions found for `n = 5` by `search.py` and sorts them into affine-equivalence classes, by applying every invertible `5×5` matrix over `GF(2)` to each function and checking (via binary search against the canonical-form list) which other optimal functions it maps to. Used to count the true number of inequivalent optimal functions, written to `optimal5.csv`.

- **[optimal_sums.py](optimal_sums.py)** — Large-scale search for optimal binomial functions `f(x) = x^d1 + α^i·x^d2` over `n` = 4–16. Uses known affine equivalences to reduce the search space, and multiprocessing to test functions in parallel. Depending on `mode` in `config.ini`, either records only the optimal functions (`optimal_sums.csv`) or the full uniformity table for the field (`<n> (sums).csv`).

- **[mults.py](mults.py)** — For all known optimal power/binomial functions (`n` = 4–15, see `get_exponents` in `functions.py`), computes each function's **multiplicity spectrum**: for every 2D subspace, the second-order derivative's kernel is computed and collisions between subspaces mapping to the same kernel are counted. The resulting histogram is written to `mults.csv`.

- **[grassmann.py](grassmann.py)** — For the same known optimal functions (`n` = 5–13), builds the **orthoderivative map** `π` sending each 2D subspace to the kernel of its derivative matrix, then tests whether `π` is a graph automorphism of the Grassmann graph on 2D subspaces (adjacency = sharing exactly one non-zero element). Saves the number and profile of automorphism violations to `grassmann.txt`.

- **[infinite.py](infinite.py)** — Investigates an infinite family of trace-based functions `f(x) = x·Tr(a·x^(2^j + 2^2j))` over `GF(2^n)` for `n = 2p` with `p` prime (`n` ∈ {6, 10, 14, 22}). Tests each candidate `j` for optimal second-order differential uniformity and, for optimal ones, computes their multiplicity spectrum (reusing `calculate_mult` from `mults.py`). Supports multiprocessing (`multi_processing` in `config.ini`) and writes results to `optimal_infinites.csv`.

- **[config.ini](config.ini)** — Central configuration read by every script at startup: the range of field sizes (`min`/`max`) to test, `mode` for `optimal_sums.py` (`A` = record all results, `P` = optimal only), and `multi_processing` for `infinite.py`.

## Installation

1. Install Python 3 (a recent 3.x release).

2. From the project directory, install the required third-party packages:

   ```bash
   pip install -r requirements.txt
   ```

   This installs `numpy`, `numba` (JIT compilation used throughout for performance), `numba-progress`, and `tqdm` (progress bars). Everything else the scripts import (`configparser`, `csv`, `os`, `math`, `multiprocessing`, `concurrent.futures`) is part of the Python standard library.

3. Edit [config.ini](config.ini) to set the field-size range (and mode) for whichever script you want to run, then run it directly, e.g.:

   ```bash
   python search.py
   ```