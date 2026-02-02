# Divergence & Absorption Engine Specification

**PHASE 1.75 — COORDINATE COMPARISON & ABSORPTION LOGIC**

> "Divergence is when present momentum (aggressors) and historical levels (passive)  
> show opposite signs. Full absorption → reversal or continuation."

---

## 1. CORE DEFINITIONS

### 1.1 Divergence

**Definition:**
Divergence occurs when present market momentum (aggressors) and historical/passive levels show **opposite charge signs**.

**Components:**
- **Aggressors:** Current trend participants (present momentum)
- **Passive:** Historical highs/lows (resting/pending orders)

**Detection:**
```
Passive:   (W−, D−, S−)
Aggressor: (W+, D+, S+)
           ↑   ↑   ↑
        All opposite signs = DIVERGENCE
```

**Integration with PHASE 1.5:**
- Passive coordinates = ChargedLevel from past formation
- Aggressor coordinates = Current ParticipantEngine state
- Comparison = CoordinateEngine.get_divergence_tfs()

---

## 2. EXHAUSTION ABSORPTION

### 2.1 Definition

**Exhaustion Absorption** occurs when passive liquidity absorbs current trend's aggressors, limiting their effect and causing reversal.

**Rule:**
```
IF passive_strength > aggressor_strength:
    → Reversal (Exhaustion Absorption)
ELSE:
    → Continuation (Imbalance)
```

### 2.2 Mechanism

1. Identify AOI (Area of Interest) with liquidity
2. Observe if aggressor moves diverge against passive coordinates
3. If fully diverged (D+/D−, S+/S−) → AOI triggers reversal

**Example:**
```
AOI Low: (D−, S−)
Present: (D+, S+)
         ↓
If fully absorbed → reversal towards AOI high
```

---

## 3. INTERNAL VS EXTERNAL ABSORPTION

### 3.1 External Absorption (External Divergence)

**Definition:** Price absorbs to **continue the passive trend**

**Target:** External liquidity that the low/high is leading to

**Example:**
```
Low leads to High
↓
Absorption at low = EXTERNAL
↓
Target = External High (reversal)
```

**Characteristic:**
- Divergence happens AGAINST external liquidity
- Signals trend REVERSAL
- Price moves toward external target

### 3.2 Internal Absorption (Internal Divergence)

**Definition:** Price absorbs to **continue the aggressive trend**

**Target:** Internal liquidity within current trend

**Example:**
```
Low creates high INSIDE aggressive move
↓
Absorption at low = INTERNAL
↓
Target = Internal High (continuation)
```

**Characteristic:**
- Divergence happens WITHIN current trend
- Signals trend CONTINUATION
- Price targets internal levels first

### 3.3 Key Distinction

```
Internal Divergence → Trend Continuation
External Divergence → Trend Reversal
```

---

## 4. FULL ABSORPTION / DIVERGENCE VALIDATION

### 4.1 Definition

**Full Absorption** occurs when **all coordinates** of passive and aggressor completely diverge.

**Validation:**
```python
def is_full_absorption(passive: Coordinate, aggressor: Coordinate) -> bool:
    """Check if all active TFs show opposite charges."""
    for tf in passive.active_tfs:
        passive_charge = getattr(passive, tf)
        aggressor_charge = getattr(aggressor, tf)
        
        if passive_charge is None or aggressor_charge is None:
            continue
        
        # Charges must be opposite (+/− or −/+)
        if passive_charge * aggressor_charge >= 0:  # Same sign or zero
            return False
    
    return True
```

### 4.2 Signal Types

**Internal Full Absorption:**
- All coordinates diverge within trend
- Signals potential pullback in trend direction
- Price targets internal levels

**External Full Absorption:**
- All coordinates diverge against external liquidity
- Signals potential reversal
- Price targets external levels

---

## 5. AOI (AREA OF INTEREST) VALIDATION

### 5.1 AOI Selection Rules

**Rule 1:** AOI is validated by **which area price is being called toward**

**Rule 2:** Highest volume levels reveal divergence patterns

**Rule 3:** Session differences matter (Frankfurt low vs London high)

**Rule 4:** Price gravitates to AOIs with **unmitigated liquidity**

### 5.2 AOI Characteristics

```python
@dataclass
class AOI:
    """Area of Interest with liquidity validation."""
    coordinate: Coordinate          # Passive coordinate at AOI
    liquidity_type: str            # "internal" or "external"
    is_mitigated: bool             # Has price returned to this level?
    volume: float                  # Volume at formation
    session: str                   # "Frankfurt", "London", "NY", etc.
    
    target_coordinate: Optional[Coordinate] = None  # What it's targeting
```

### 5.3 Multi-Session Absorption

**Critical Rule:**
Full absorption can happen **across multiple sessions**.

**Example:**
```
Frankfurt Low (D−, S−)
    ↓ Internal Absorption
London High (D+, S+)
    ↓ External Absorption
NY Reversal → External liquidity target
```

---

## 6. ABSORPTION STRENGTH CALCULATION

### 6.1 Divergence Score

```python
def calculate_divergence_score(
    passive: Coordinate,
    aggressor: Coordinate
) -> float:
    """
    Calculate divergence strength (0.0 - 1.0).
    
    1.0 = Full divergence (all TFs opposite)
    0.0 = No divergence (all TFs same)
    """
    divergent_tfs = passive.get_divergence_tfs(aggressor)
    comparable_tfs = count_comparable_tfs(passive, aggressor)
    
    if comparable_tfs == 0:
        return 0.0
    
    return len(divergent_tfs) / comparable_tfs
```

### 6.2 Absorption Type Detection

```python
def detect_absorption_type(
    aoi: AOI,
    current_coordinate: Coordinate,
    external_target: Optional[Coordinate]
) -> str:
    """
    Determine if absorption is internal or external.
    
    Returns: "internal", "external", or "none"
    """
    divergence_score = calculate_divergence_score(
        aoi.coordinate, current_coordinate
    )
    
    if divergence_score < 0.5:
        return "none"  # Not enough divergence
    
    # Check if targeting external liquidity
    if external_target:
        # Price moving toward external target = external absorption
        if is_moving_toward(current_coordinate, external_target):
            return "external"
    
    # Otherwise, internal absorption (trend continuation)
    return "internal"
```

---

## 7. PRACTICAL EXAMPLES

### 7.1 Frankfurt Low → London High

**Scenario:**
```
Frankfurt Low: (D−, S−)
London Open:   (D+, S+) [Buy signal]
```

**Analysis:**
1. London opens as buyer
2. Frankfurt low creates **internal divergence**
3. Internal absorption → price moves up
4. Target: Internal high (continuation)

**Code:**
```python
frankfurt_low = Coordinate(price=1.0950, D=-1, S=-1)
london_open = Coordinate(price=1.1000, D=+1, S=+1)

divergence = frankfurt_low.get_divergence_tfs(london_open)
# → ['D', 'S'] (full divergence)

absorption_type = detect_absorption_type(
    aoi=AOI(coordinate=frankfurt_low, liquidity_type="internal"),
    current_coordinate=london_open,
    external_target=None
)
# → "internal" (trend continuation)
```

### 7.2 External Reversal Example

**Scenario:**
```
AOI Low:  (W−, D−, S−)
Present:  (W+, D+, S+)
Target:   External High (W+, D+) from previous day
```

**Analysis:**
1. Full divergence at AOI low
2. External liquidity target exists (previous high)
3. External absorption → reversal signal
4. Target: External high

---

## 8. INTEGRATION WITH HORC STACK

### 8.1 Updated Decision Hierarchy

```
ParticipantEngine (PHASE 1 - WHO)
    ↓
FlipEngine (PHASE 1.5 - WHEN)
    ↓
ChargeEngine (PHASE 1.5 - +/−)
    ↓
CoordinateEngine (PHASE 1.5 - STATE VECTORS)
    ↓
DivergenceEngine (PHASE 1.75 - DIVERGENCE DETECTION) ← NEW
    ↓
AbsorptionEngine (PHASE 1.75 - ABSORPTION LOGIC) ← NEW
    ↓
Opposition (eligibility)
    ↓
Quadrant (authority)
    ↓
Imbalance/Liquidity (validation)
```

### 8.2 Data Flow

```
1. ChargeEngine assigns charges to levels
2. CoordinateEngine builds state vectors
3. DivergenceEngine compares passive vs aggressor
4. AbsorptionEngine determines internal vs external
5. Signal generation based on absorption type
```

---

## 9. KEY CONCEPTS (DOCTRINE)

### Rule 1: Divergence Detection
> "Divergence = present vs passive momentum opposite signs"

### Rule 2: Exhaustion Absorption
> "Passive overwhelms aggressor → reversal"

### Rule 3: Internal Absorption
> "Trend continuation within aggressive move"

### Rule 4: External Absorption
> "Trend reversal toward external liquidity"

### Rule 5: AOI Validation
> "AOI selection depends on which liquidity is calling price"

### Rule 6: Multi-Session Tracking
> "Full absorption may span multiple sessions"

---

## 10. IMPLEMENTATION REQUIREMENTS

### 10.1 DivergenceEngine

**Purpose:** Compare passive and aggressor coordinates

**Key Methods:**
- `calculate_divergence_score(passive, aggressor) -> float`
- `is_full_divergence(passive, aggressor) -> bool`
- `get_divergent_timeframes(passive, aggressor) -> List[str]`

### 10.2 AbsorptionEngine

**Purpose:** Determine absorption type and strength

**Key Methods:**
- `detect_absorption_type(aoi, current, target) -> str`
- `calculate_absorption_strength(passive, aggressor) -> float`
- `identify_target(aoi, current) -> Coordinate`

### 10.3 AOI Manager

**Purpose:** Track and validate Areas of Interest

**Key Methods:**
- `register_aoi(coordinate, liquidity_type, session) -> AOI`
- `is_mitigated(aoi, current_price) -> bool`
- `get_active_aois(session) -> List[AOI]`

---

## 11. TESTING REQUIREMENTS

### Test Cases:

1. ✅ Full divergence detection (all TFs opposite)
2. ✅ Partial divergence detection (some TFs opposite)
3. ✅ Internal absorption identification
4. ✅ External absorption identification
5. ✅ Multi-session absorption tracking
6. ✅ AOI mitigation validation
7. ✅ Absorption strength calculation

---

## 12. VISUAL REPRESENTATION

```
INTERNAL ABSORPTION (Continuation)
═══════════════════════════════════
     External High (target)
            ↑
            │
    Internal High (D+, S+)
            ↑
            │ [Internal Absorption]
            │
    AOI Low (D−, S−) ← Passive
            │
    [Aggressor: D+, S+]
            │
            ↓
     (Trend continues up)


EXTERNAL ABSORPTION (Reversal)
═══════════════════════════════════
     External High (W+, D+) ← Target
            ↑
            │ [External Absorption]
            │
    Present (W+, D+, S+) ← Aggressor
            │
            ↓
    AOI Low (W−, D−, S−) ← Passive
            │
            ↓
     (Reversal toward external high)
```

---

## 13. NEXT STEPS

**Immediate Implementation:**
1. Create `DivergenceEngine` class
2. Create `AbsorptionEngine` class
3. Create `AOI` management system
4. Write comprehensive tests
5. Integrate with existing coordinate system

**Deliverables:**
- `src/core/divergence_engine.py`
- `src/core/absorption_engine.py`
- `src/core/aoi_manager.py`
- `tests/test_divergence_absorption.py`

---

## 14. CRITICAL INSIGHTS

### Naked Eye Analysis Limitation
> "Visual structure alone is not enough → use divergence and absorption  
> to track real liquidity movement."

### Dynamic Absorption
> "Multiple AOIs may absorb separately. Price may revisit external liquidity  
> even after internal divergence."

### Session-Level Tracking
> "Frankfurt low vs London high matters. Track sessional and weekly volume."

### Coordinate Foundation
> "Everything depends on accurate coordinate building (PHASE 1.5).  
> Without immutable charges, divergence detection fails."

---

**Status:** Specification complete. Ready for implementation.

**Foundation:** PHASE 1.5 Coordinate Engine provides all necessary primitives.

**Next:** Build divergence detection + absorption logic engines.
