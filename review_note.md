Pricer is pure numerical algebra: it receives \(p_0\), \(A\), \(q\), bounds and \(B\).
It has no products, dates, EWMA, IPCA, pandas or Pandera concepts.
The orchestration layer owns market data, monthly delivery profiles and exposures.
EHUB has three typed operations: normalized tickers, deals and latest tenor prices.
Nested ticker `features_` are normalized once and pivoted into ordinary columns.
`CurveGranularity` defaults to M/Q/S/A and accepts interval overrides.
Product boundaries induce non-overlapping atomic strips, defining \(p=Bx\).
Selection is performed on the economic matrix \(AB\), not on ticker names.
All dual raw/indexed marks enter one weighted least-squares objective.
Annual reliability is a precision weight, not an exact equality.
Monthly, quarterly, semiannual and annual rows may be redundant without conflict.
Inconsistent marks produce the minimum weighted residual; exact mode is audit-only.
IPCA and energy weights are only coefficients of economic exposure.
IPCA is frozen at each selected block head before the joint raw/indexed solve.
Vector prices solve one curve; product-by-scenario matrices solve several curves.
Only the liquid segment through the market cutoff is projected.
The same raw state minimizes raw and indexed errors under monthly floors and caps.
DCIDE is appended raw; one vectorized multiplication indexes the complete curve.
The default deterministic soft solution is idempotent and preserves the illiquid segment.
Pandera validates tables; dataclasses exist only for stateful policies, services or results.
All 147 tests pass with 100% measured line coverage.
