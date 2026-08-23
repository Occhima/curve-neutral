# Soft dual neutralization and infeasibility

Let the raw monthly curve be \(p=Bx\), where \(B\) contains the chosen atomic
blocks and seasonal shape. Normal curve construction solves

\[
\min_x\;
\lVert A_{raw}Bx-q_{raw}\rVert^2_{W_r}+
\lVert A_{indexed}Bx-q_{indexed}\rVert^2_{W_i},
\qquad \ell\leq Bx\leq u.
\]

`A_indexed` already contains energy weights, discounting and monthly IPCA
factors. Therefore the numerical core only sees one stacked least-squares
problem plus bounds. Inconsistent raw, indexed or overlapping product marks do
not make this problem infeasible: they create nonzero optimal residuals.

Only hard restrictions can make the recommended problem infeasible:

1. **Block/bound conflict.** A shared block or fixed seasonal ratio in \(B\)
   can require the same latent price to be simultaneously above one month's
   floor and below another month's cap.
2. **Invalid bound intersection.** Raw bounds and any converted indexed bounds
   may have an empty intersection.
3. **Optional hard intervals.** Bid/ask regions or deliberately exact audit
   rows can have an empty intersection with monthly bounds.

The optional `build_exact_dual_anchor_plan` audit mode can additionally fail
when dependent rows have inconsistent targets, equivalently
`rank(M) < rank([M | q])`. This exact-mode failure is not part of normal curve
construction.

The prior curve, smoothness penalty, anchor precision and surface weights cannot
create infeasibility: they only move the optimum inside the feasible set. DCIDE
concatenation and final IPCA multiplication happen after the liquid solve and do
not change its feasible set. Each scenario is independent.

NaNs, wrong dimensions, duplicated labels and `floor > cap` are invalid input,
not market infeasibility. These belong at the Pandera/array boundary. All hard
contradictions are left to the single linear feasibility phase in Pricer; no
product-specific guard clauses are required. Solver `tolerance` controls
numerical feasibility, not acceptable economic pricing error; the latter is
expressed by objective weights or explicit price bands.

References: [SciPy/HiGHS linear feasibility model](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html),
[Fleten and Lemming (2003)](https://doi.org/10.1016/S0140-9883(03)00039-2),
and [Benth, Koekebakker and Ollmar (2007)](https://doi.org/10.3905/jod.2007.694791).
