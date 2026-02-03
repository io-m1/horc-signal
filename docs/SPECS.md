## Technical Specifications

### Flip Engine

A Flip Point marks when participant control changes on a given timeframe.

Temporal Constraint: A flip is valid only before the next corresponding open.

Corresponding Opens:
- Session to next session open
- Day to next day open
- Week to next week open
- Month to next month open

Flip States:
- ACTIVE: Participant in control, no flip yet
- PENDING: Opposition detected, flip not confirmed
- CONFIRMED: Flip confirmed, within validity window
- LOCKED: Next open reached, state immutable
- INVALID: Flip invalidated

### Charge Engine

Any high or low inherits the participant state active at the time it was formed.

Charge Assignment:
- BUYER control: + (positive charge)
- SELLER control: - (negative charge)
- No control: 0 (neutral)

Charge Per Timeframe:
- Session: S+ / S-
- Day: D+ / D-
- Week: W+ / W-
- Month: M+ / M-

### Coordinate Engine

A coordinate is the ordered participant state of a price level across valid timeframes.

Format: (M±, W±, D±, S±)

Example: (M-, W+, D+, S-)
- Monthly context: seller-born
- Weekly context: buyer-born
- Daily context: buyer-born
- Session context: seller-born

Only timeframes that exist at the moment of formation are included.

### Divergence Detection

Divergence occurs when present market momentum (aggressors) and historical/passive levels show opposite charge signs.

Full Divergence: All coordinates of passive and aggressor completely diverge.

Internal Divergence: Happens within current trend, signals continuation.
External Divergence: Happens against external liquidity, signals reversal.

### Absorption Engine

Exhaustion Absorption occurs when passive liquidity absorbs current trend's aggressors.

Rule:
- If passive_strength > aggressor_strength: Reversal
- Else: Continuation

Absorption Strength = divergence_score * (1 - distance_to_aoi) * volume_weight

### AOI (Area of Interest)

Properties:
- coordinate: Passive coordinate at AOI
- liquidity_type: internal or external
- is_mitigated: Has price returned to this level
- volume: Volume at formation
- session: Frankfurt, London, NY, etc.

### Liquidity Engineering (Experimental)

Tested concepts from iSpeculatefx methodology. Results showed these filters hurt accuracy when systematized.

AOL (Accumulation of Liquidity): Counts untapped liquidity pools at swing highs/lows.
ELQ (Extreme Liquidity Quotient): Ratio of extreme swing accumulation.
VV Analysis (Void/Volume): Detects volume voids that attract price.

Testing showed baseline HORC (39.6% WR) outperformed all LE-enhanced configurations (29-38% WR).

Conclusion: LE concepts work as discretionary overlays, not mechanical filters.
