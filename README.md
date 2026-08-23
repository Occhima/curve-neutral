# Arbitrage-free curve refactor

This bundle is split by ownership. Copy `pricer/` into the Pricer repository
and `orchestration/` into the flow repository.

| Project | Owns | Must not own |
| --- | --- | --- |
| Pricer | arrays, linear restrictions, convex projection, basis matrices | pandas, products, dates, tickers, BBCE/EHUB, EWMA, IPCA policy |
| Orchestration | BBCE/EHUB access, DataFrames, Pandera, price selection, anchor estimation, economic exposures | optimization logic |

Production classes are frozen dataclasses that carry related state or Pandera
dataframe contracts. Stateless behavior remains in functions.

## Pricer API

```python
from pricer.curves.arbitrage import LinearObservations, solve_curve

observations = LinearObservations.exact(exposure_matrix, anchor_prices)
solution = solve_curve(unbalanced_curve, observations)
balanced_curve = solution.prices
```

`anchor_prices` may be `(anchors,)` or `(anchors, scenarios)`. Exact, soft and
bid/ask observations share the same linear representation. The optional basis
maps latent block prices into monthly prices and can preserve a fixed seasonal
shape. Product frequency is therefore not solver logic: monthly, quarterly,
semiannual and annual prices are only rows of the exposure matrix.

## Automatic anchor basis

The orchestration layer can turn all eligible product marks into the smallest
exact system needed by the solver. It does not infer semantics from ticker
text: each product is described only by its monthly delivery profile.

```python
from curve_orchestration import build_anchor_plan, neutralize_anchor_plan

plan = build_anchor_plan(
    unbalanced_curve,
    delivery_profiles,  # rows=product_id, columns=monthly tenors
    anchor_prices,  # Series or product x scenario DataFrame
    quality=market.weight_series(),
    mandatory=annual_product_ids,
    reconcile=True,
)
result = neutralize_anchor_plan(plan)
```

The union of all delivery boundaries creates contiguous atomic strips. The
algorithm then evaluates the economic matrix \(AB\), retains mandatory annual
products first and admits the remaining products in descending quality only
when they increase matrix rank. Therefore monthly, quarterly and semiannual
marks add shape information without duplicating an annual equality. Redundant
marks remain in `plan.diagnostics`; with `reconcile=True`, they influence a
weighted least-squares pre-fit while mandatory annual prices stay exact.

Only months covered by a candidate product enter the projection. All other
months are copied exactly, so one inconsistent year cannot move another year.
The final exact projection is idempotent: applying the same plan to its own
result returns the same curve. `plan.decisions` explains every selected or
redundant product and `plan.diagnostics` reports market, reconciled and implied
prices for every scenario.

## Market cutoff, DCIDE and one-time indexation

The complete service solves only the liquid block. Both raw and indexed marks
constrain the same latent raw state; DCIDE is appended afterward and the IPCA
factor is multiplied only once, over the complete monthly curve.

```python
from curve_orchestration import build_dual_anchor_plan, neutralize_forward_curve

plan = build_dual_anchor_plan(
    unbalanced_curve,
    delivery_profiles,
    raw_anchor_prices,
    indexed_anchor_prices,
    quality=anchor_precision_by_product,
    raw_weight=1.0,
    indexed_weight=1.0,
    cutoff=market_cutoff,
)
forward = neutralize_forward_curve(plan, dcide_curve)

raw_curve = forward.wide_raw_curve()
indexed_curve = forward.wide_indexed_curve()
```

The default dual plan is one constrained weighted least-squares problem:

\[
\min_x\;
\sum_j w_{r,j}(A_{r,j}Bx-q_{r,j})^2+
\sum_j w_{i,j}(A_{i,j}Bx-q_{i,j})^2,
\qquad floor\leq Bx\leq cap.
\]

Every raw and indexed mark enters the same solve. Zero residual is preferred
when possible; inconsistent marks produce the minimum weighted compromise
instead of an exception. `quality` should represent relative precision, ideally
inverse error variance. `raw_weight` and `indexed_weight` control the relative
importance of the two surfaces. `build_exact_dual_anchor_plan` remains only as
an explicit audit mode.

Months after `cutoff` never enter the optimization. `dcide_curve` supplies
their raw `price` and `index_factor`. Monthly `floor` and `cap` columns are hard
bounds on the raw market curve and are passed directly to Pricer. If bounds
refer instead to indexed prices, orchestration must first convert them to raw
bounds by dividing by the positive monthly index factor.

## Curve granularity and the unbalanced market curve

`CurveGranularity` is injected exactly where market products become monthly
tenors. Its default priority is monthly, quarterly, semiannual, then annual.
An override replaces that priority inside an inclusive interval and does not
silently fall back to another frequency.

```python
from curve_orchestration import CurveGranularity

granularity = CurveGranularity(
    overrides=(("2027-01", "2027-12", "annual"),),
)
```

With no override, a market containing January monthly, Q1 quarterly, H1
semiannual and CAL27 annual products selects M for January, Q for February and
March, S for April through June, and A for July through December. The override
above selects CAL27 for every 2027 tenor.

`build_unbalanced_curve` copies the selected product price into each monthly
tenor. A contiguous run of the same product is one delivery block, and its
monthly `index_factor` is frozen at the block's first month. Thus a Q1/residual
split uses January's factor for Q1 and April's factor for April through
December.

## BBCE/EHUB market-price service

`EhubMarketService` has exactly three public operations, all decorated with
`@pandera.check_types`:

1. `get_negotiable_tickers` fetches tickers and uses `json_normalize` plus
   `pivot` to turn each nested `features_` name/value pair into a column;
2. `get_deals` fetches January 1 through the supplied reference date;
3. `get_latest_prices` filters eligible deals, keeps the freshest product mark
   and applies the injected `CurveGranularity` through the market cutoff.

`EhubClient.login` implements the documented login boundary, while an already
authenticated `httpx.Client` may still be injected by the container. It calls:

- `POST /bus/v2/login`;
- `GET /bus/v1/negotiable-tickers?walletId=...`;
- `GET /bus/v2/all-deals/report?initialPeriod=...&finalPeriod=...`.

The second path includes `/report`. The endpoint and parameter names are frozen
configuration, so an environment-specific variation does not enter the
statistical functions.

```python
import httpx
import pandas as pd
from curve_orchestration import (
    AnchorPolicy,
    CurveGranularity,
    EhubClient,
    EhubCredentials,
    EhubMarketService,
    build_unbalanced_curve,
    estimate_anchor_prices,
    neutralize_market_curve,
)

credentials = EhubCredentials(company_code, email, password, api_key)
reference_date = pd.Timestamp("2026-08-22 18:00", tz="America/Sao_Paulo")
cutoff = pd.Timestamp("2028-12-31")
granularity = CurveGranularity(
    overrides=(("2027-01", "2027-12", "annual"),),
)

with httpx.Client(base_url=bbce_environment_url, timeout=30.0) as http:
    service = EhubMarketService(EhubClient.login(http, credentials))
    tickers = service.get_negotiable_tickers(wallet_id)
    deals = service.get_deals(reference_date)
    latest = service.get_latest_prices(
        tickers,
        deals,
        curve[["tenor"]],
        reference_date=reference_date,
        cutoff=cutoff,
        granularity=granularity,
    )
    unbalanced = build_unbalanced_curve(curve, latest)
    market = estimate_anchor_prices(
        tickers[["product_id", "description"]],
        deals[["deal_id", "product_id", "price", "quantity", "traded_at"]],
        reference_date,
        policy=AnchorPolicy(half_life=pd.Timedelta("3D")),
    )

forward = neutralize_market_curve(
    curve,
    tickers,
    market.price_series(),
    market.anchors.set_index("product_id")["last_inlier_at"],
    dcide_curve,
    reference_date=reference_date,
    cutoff=cutoff,
    granularity=granularity,
    quality=market.weight_series(),
)
```

The environment host is configuration; the `/bus` routes are not duplicated in
flow code. The client reads the documented `tickers` envelope and follows every
page reported by `x-number-of-pages`, sending the documented `page` header.
`EhubFields` isolates the JSON names. Ticker `id` joins deal `productId`, while
prices and timestamps come from `unitPrice` and `createdAt`. The normalizer does
not parse product descriptions.

The literal latest price remains separate from the robust MAD/EWMA estimator.
The latter supplies product-level anchor marks and precision weights; the
granularity policy maps those marks to monthly tenors. Annual products remain
anchors even when every month selected a finer product.

## Raw and indexed no-arbitrage

Use two explicit target surfaces when both the raw curve and the IPCA-adjusted
curve must reproduce their anchors:

```python
anchors = build_raw_and_indexed_anchor_matrix(
    unbalanced_curve,  # contains the monthly index_factor
    delivery_profiles,
    raw_anchor_prices,
    indexed_anchor_prices,
)
result = neutralize_curve(unbalanced_curve, anchors)
```

The low-level exact adapter stacks the systems

\[
A_{raw}p=q_{raw},\qquad A_{indexed}p=q_{indexed}.
\]

Rows are labelled `raw:<product_id>` and `indexed:<product_id>` in the
diagnostics. Both price arguments may be Series or scenario DataFrames. The
Pricer still receives only one numerical matrix. If prices, factors, bounds and
the selected block basis cannot satisfy both systems, exact neutralization
raises `InfeasibleCurveError`; it never hides the conflict behind a compromise.
Normal curve construction uses `build_dual_anchor_plan` instead, so both rows
are soft observations in one constrained least-squares objective.

## Statistical rule

For each product, duplicated deal IDs, non-negotiable tickers, rejected
statuses and observations after the cutoff are removed. Let

\[
m=\operatorname{median}(P),\qquad
MAD=\operatorname{median}(|P-m|).
\]

A price is retained when
\(|P-m|\leq c\,1.4826\,MAD\). When `MAD == 0`, only observations equal to the
median survive. The anchor is then

\[
\widehat P=\frac{\sum_j w_jP_j}{\sum_jw_j},\qquad
w_j=2^{-(t_c-t_j)/h},
\]

with optional multiplication by quantity. This is the time-aware EWMA written
as a vectorized weighted mean. The output contains `product_id`, `ewma_price`,
last inlier price/time, median, MAD, counts, age and effective weight.

The estimator does not silently declare the result exact. Use exact anchors
only when equality is economically intended; independent asynchronous EWMA
marks are usually better represented as soft or bid/ask observations. A prior
curve is mathematically irrelevant only when the exact anchor matrix identifies
every latent block; otherwise the initial curve still selects the unresolved
degrees of freedom.

## Verification

```bash
uv run pytest -m unit
uv run pytest -m functional
uv run pytest --cov --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
```

The 147 tests cover every public contract and production statement, including
the HTTP boundary, configurable payload fields, status/universe filtering,
deal deduplication, zero-MAD behavior, recency and quantity weights, empty
surfaces, matrix scenarios, dual raw/indexed equality, incompatibility, bounds,
indexation, cutoff isolation, DCIDE composition, atomic strips, rank-based
anchor selection, weighted dual compromises, surface precision, price bounds,
architectural purity, cross-year locality, identity, idempotence, nested
feature pivoting, the three-operation EHUB boundary, granularity overrides,
annual-anchor retention, IPCA block heads, and the complete pipeline.
