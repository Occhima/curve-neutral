# Arbitrage-free forward curve

Split by ownership. Copy `pricer/` into the Pricer repository and
`orchestration/` into the flow repository.

| Project | Owns | Must not own |
| --- | --- | --- |
| Pricer | arrays, linear restrictions, convex projection, basis matrices | pandas, products, dates, tickers, EHUB, EWMA, IPCA |
| Orchestration | EHUB access, DataFrames, Pandera, price selection, anchor estimation, exposures | optimization logic |

Orchestration is three modules:

| Module | Contents |
| --- | --- |
| `ehub.py` | HTTP client, the three typed service operations, `select_market_universe`, anchor pricing |
| `curve.py` | `CurveGranularity`, tenor pricing, unbalanced curve, anchor plan, DCIDE splice |
| `neutralization.py` | the pandas↔numpy boundary: exposure matrix in, Pricer solution out |

## The pipeline

```bash
uv run python examples/build_forward_curve.py
```

`examples/build_forward_curve.py` runs the whole thing offline against a fake
client. The five steps:

**1. EHUB.** Two API reads plus their inner join, then the tradeable universe.

```python
service = EhubMarketService(EhubClient.login(http, credentials))
tickers = service.get_negotiable_tickers(wallet_id)  # features_ pivoted to columns
deals = service.get_deals(reference_date)  # Jan 1 -> reference date
universe = select_market_universe(  # SE / CON / FIX
    service.get_market_deals(tickers, deals)
)
```

All three operations carry `@pandera.check_types`. `get_negotiable_tickers`
uses `json_normalize` plus `pivot`, so every nested `features_` name/value pair
becomes an ordinary column: `submarket`, `source`, `price_type`, `start`, `end`.
`EhubFields` isolates the provider's JSON names.

**2. Anchor price.** One fair price per product. Prices within
`|P - median| <= threshold * 1.4826 * MAD` survive; when `MAD == 0` only prices
equal to the median do. The anchor is the weighted mean with
`w = 2 ** (-age / half_life)`, whose sum is `effective_weight`.

The solver's anchor weight is `precision = effective_weight / (1.4826 * MAD)**2`,
an inverse-variance estimate. Recency mass alone would let two fresh quotes 40
apart outweigh a tight stack: how much recent evidence exists is not how much it
agrees. Products with `MAD == 0` borrow the median robust scale of the others
rather than claiming infinite confidence.

The median is decay-weighted so the centre tracks where the market is *now* —
against a flat yearly median a genuine repricing is rejected as an outlier and
the anchor stays stale for months. The MAD is deliberately **not** weighted: a
decay-weighted median puts most of the weight on one observation, which drives
the MAD to zero and turns the filter into an equality test that throws away
ordinary quotes. Location needs recency; dispersion needs the sample.

```python
quotes = estimate_anchor_prices(universe, reference_date, policy=AnchorPolicy())
```

**3. Plan.** One product per tenor, IPCA frozen per block, anchors stated.

```python
cutoff = market_cutoff(reference_date)  # Dec 31 of the second year ahead
plan = build_market_plan(
    curve,
    quotes,
    cutoff=cutoff,
    granularity=CurveGranularity().for_year(2028, ("ANU",)),
)
```

`CurveGranularity` defaults to the liquidity order `("MEN", "TRI", "SEM",
"ANU")`. A rule replaces that order inside its months, and a one-element
priority such as `("ANU",)` has no fallback — that is what makes "price all of
2028 off CAL28" deterministic. `for_year` and `for_months` return a new policy.

A contiguous run of the same product is one delivery block, and its monthly
`index_factor` is frozen at the block's first month. A Q1/residual split
therefore reindexes Q1 by January's factor and the residual by April's.

**4. Solve.** The market block only. DCIDE is not an input here.

```python
neutralized = neutralize_anchor_plan(plan)
```

One raw state `p = Bx` is fitted so that the raw price *and* the IPCA-corrected
price are both as close as possible to their anchors:

```
min_x  Σ w_r (A_r B x − q_r)² + Σ w_i (A_i B x − q_i)²    s.t.  floor ≤ Bx ≤ cap
```

Inconsistent marks do not raise: they produce the minimum weighted residual.
Only hard restrictions (block/bound conflicts, empty bound intersections) can
make the problem infeasible — see `infeasibility.md`.

**5. Compose.** Now, and only now, DCIDE and indexation.

```python
forward = finalize_forward_curve(plan, neutralized, dcide_curve)
raw = wide_curve(forward, "raw_price")
indexed = wide_curve(forward, "indexed_price")
```

Months after `cutoff` take their raw price and factor from `dcide_curve`. The
IPCA factor multiplies the concatenated raw curve exactly once.

## Pricer API

```python
from pricer.curves.arbitrage import LinearObservations, solve_curve

solution = solve_curve(unbalanced_curve, LinearObservations.exact(exposure, prices))
```

`prices` may be `(anchors,)` or `(anchors, scenarios)`; scenarios solve
independently against the same structure. Exact, soft and bid/ask observations
share one linear representation. Product frequency is not solver logic —
monthly, quarterly, semiannual and annual prices are only rows of the exposure
matrix.

## Verification

```bash
uv run pytest -m unit
uv run pytest -m functional
uv run pytest --cov --cov-report=term-missing   # 100% enforced
uv run ruff check . && uv run ruff format --check .
```
