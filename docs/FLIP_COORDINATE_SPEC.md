# Flip + Coordinate Engine Specification

**PHASE 1.5 — TEMPORAL FINALITY & STATE ENCODING**

> "A flip is valid **only before the next corresponding open**.  
> Whichever participant is dominant at overlap time becomes authoritative."

---

## 1. FLIP ENGINE — TEMPORAL CONTROL

### 1.1 Definition

A **Flip Point (FP)** marks when participant control CHANGES on a given timeframe.

**Temporal Constraint:**
- A flip is valid **only before the next corresponding open**
- Once next open overlaps → flip is **invalid forever**

**Corresponding Opens:**
- Session ↔ next session open
- Day ↔ next day open
- Week ↔ next week open
- Month ↔ next month open

### 1.2 Flip Detection Algorithm

For a given timeframe `TF`:

1. **Identify TF open** (session/day/week/month boundary)
2. **Observe which side swept first** (high or low)
   - High swept first → BUYER active
   - Low swept first → SELLER active
3. **3-cycle allowance** (OR logic):
   - Participant has 3 candles to establish control
4. **Opposition detection** (from PHASE 1):
   - If opposite side swept **before next TF open** → FP registered
   - After next TF open → FP locked or discarded
5. **State finality**:
   - Whichever participant is dominant at next open = authoritative

### 1.3 Flip Validity Window

```
Current TF Open ───────────────────> Next TF Open
     │                                    │
     │◄────── Flip Valid Window ─────────►│
     │                                    │
   [Active]                          [Locked]
```

**Before next open:** Flip can be registered/invalidated  
**After next open:** Flip is **immutable**

### 1.4 Flip States

```python
FlipState {
    ACTIVE,         # Participant in control, no flip yet
    PENDING,        # Opposition detected, flip not confirmed
    CONFIRMED,      # Flip confirmed, within validity window
    LOCKED,         # Next open reached, state immutable
    INVALID         # Flip invalidated (no opposition by next open)
}
```

---

## 2. PARTICIPANT CHARGE ENGINE (+/− LABELING)

### 2.1 Definition

> "Any high or low **inherits the participant state active at the time it was formed**."

**Charge Assignment:**
- BUYER control → `+` (positive charge)
- SELLER control → `−` (negative charge)
- No control → `0` (neutral/inconclusive)

### 2.2 Charge Inheritance Rule

```
Before FP: All highs/lows = original participant charge
After FP:  All highs/lows = new participant charge
```

**Example (Daily):**
- Day opens, BUYER active → all highs/lows = `D+`
- Flip occurs (SELLER takes control) → all NEW highs/lows = `D−`
- Previous `D+` levels remain `D+` (immutable)

### 2.3 Charge Per Timeframe

Each timeframe has independent charge state:

- Session: `S+` / `S−`
- Day: `D+` / `D−`
- Week: `W+` / `W−`
- Month: `M+` / `M−`

Charge is assigned **at formation time**, not retroactively.

---

## 3. COORDINATE ENGINE — MULTI-TF STATE VECTORS

### 3.1 Definition

A **coordinate** is the ordered participant state of a price level across valid timeframes.

**Format:**
```
(M±, W±, D±, S±)
```

**Example:**
```
(M−, W+, D+, S−)
```

Interpretation:
- Monthly context: seller-born
- Weekly context: buyer-born
- Daily context: buyer-born
- Session context: seller-born

### 3.2 Highest Volume Open (HVO) Rule

> "Only timeframes that **exist at the moment of formation** are included."

**Examples:**

| Scenario | Active TFs | Coordinate Format |
|----------|-----------|-------------------|
| Same day, same session | S | `(S±)` |
| Same day, different session | S, D | `(D±, S±)` |
| Same week, different day | S, D, W | `(W±, D±, S±)` |
| Different month | S, D, W, M | `(M±, W±, D±, S±)` |

This prevents false stacking.

### 3.3 Coordinate Immutability

Once a timeframe closes:
- Its charge state for that level is **immutable**
- No recalculation
- State is final

### 3.4 Coordinate Comparison

Coordinates enable precise targeting:

```
Target: (M−, W+, D+, S−)
Current: (M−, W+, D−, S−)
               ↑
         Daily flip changed state
```

This is NOT divergence — it's **state encoding** for precise zone identification.

---

## 4. INTEGRATION WITH HORC STACK

**Revised HORC Architecture:**

```
1. Temporal Open Engine
   └─ Defines when flips are allowed (TF boundaries)

2. Participant Engine (PHASE 1) ✅
   └─ Determines WHO is in control (BUYER/SELLER)

3. Flip Engine (PHASE 1.5) 🔥
   └─ Validates WHEN control changes + temporal finality

4. Charge Engine (PHASE 1.5) 🔥
   └─ Assigns +/− to highs/lows at formation

5. Coordinate Engine (PHASE 1.5) 🔥
   └─ Encodes multi-TF state vectors (M±, W±, D±, S±)

6. Liquidity Registration (PHASE 2)
   └─ Marks swing points with coordinates

7. Imbalance Extraction (PHASE 3)
   └─ Identifies imbalance zones

8. Relationship Validator (PHASE 4)
   └─ 6 rules + tier matching

9. HP3 Engine (PHASE 5)
   └─ Entry timing + risk compression
```

---

## 5. CRITICAL DISTINCTIONS

### Flip ≠ Signal
- Flip = state change in participant control
- Signal = actionable trade setup (comes later)

### Coordinate ≠ Entry
- Coordinate = state encoding (what exists)
- Entry = execution decision (what to do)

### Temporal Finality = Non-Negotiable
- After next open → state is **locked forever**
- No retroactive changes
- This is what makes HORC deterministic

---

## 6. DATA STRUCTURES (CONCEPTUAL)

### Level Object
```python
Level {
    price: float
    timestamp: int
    coordinates: {
        M: +1 / -1 / None
        W: +1 / -1 / None
        D: +1 / -1 / None
        S: +1 / -1 / None
    }
    flip_state: FlipState
    participant_at_formation: ParticipantType
}
```

### Flip Validator
```python
if now < next_corresponding_open(TF):
    if opposite_side_swept:
        flip_state = CONFIRMED
    else:
        flip_state = PENDING
else:
    flip_state = LOCKED  # Immutable
```

### Coordinate Assignment
```python
for tf in active_timeframes_at_formation(timestamp):
    level.coordinates[tf] = current_participant_charge(tf)
```

**No recalculation. Ever.**

---

## 7. PINE SCRIPT CONSTRAINTS

Pine Script cannot:
- Loop historical states deeply
- Store complex objects natively
- Recalculate past states freely

**Solution:**
- Bar index math for temporal checks
- Persistent `var` states for participant tracking
- Boolean guards for flip validity

**Example:**
```pinescript
var int D_state = 0  // +1 (BUYER) or -1 (SELLER)
var int W_state = 0
var int M_state = 0

// Flip updates state ONLY if before next open
if time < next_day_open:
    if opposite_swept:
        D_state := -D_state  // Flip
```

---

## 8. TESTING REQUIREMENTS

### Flip Engine Tests
1. Detect flip within validity window
2. Lock flip after next open
3. Invalidate flip if no opposition by next open
4. Handle 3-cycle allowance correctly

### Charge Engine Tests
1. Assign correct +/− at formation
2. Preserve charge after flip (immutability)
3. Independent charge per TF

### Coordinate Engine Tests
1. Build correct coordinate tuples
2. Apply HVO rule (only active TFs)
3. Compare coordinates accurately

---

## 9. DELIVERABLES

### PHASE 1.5 Outputs
1. **Flip Engine Module** (`src/core/flip_engine.py`)
2. **Charge Engine Module** (`src/core/charge_engine.py`)
3. **Coordinate Engine Module** (`src/core/coordinate_engine.py`)
4. **Integration Tests** (`tests/test_flip_coordinate.py`)

### Expected Behavior
```python
# After PHASE 1.5:
level = Level(price=1.1000, timestamp=...)

# Participant Engine (PHASE 1)
participant = engine.register_participant(...)

# Flip Engine (PHASE 1.5)
flip_state = flip_engine.validate_flip(tf, timestamp)

# Charge Engine (PHASE 1.5)
charge = charge_engine.assign_charge(tf, participant)

# Coordinate Engine (PHASE 1.5)
coordinates = coordinate_engine.build_coordinate(
    level, active_tfs=['M', 'W', 'D', 'S']
)

# Result: (M−, W+, D+, S−)
```

---

## 10. DOCTRINE SUMMARY

> **"A flip is valid only before the next corresponding open."**

> **"Any high or low inherits the participant state active at formation."**

> **"Only timeframes that exist at formation are included in coordinates."**

> **"Once a timeframe closes, its charge state is immutable."**

> **"Flip ≠ Signal. Coordinate ≠ Entry. They are state layers."**

---

**Next Step:** Implement deterministic Flip Validator module first.  
**Why:** Everything depends on flip legitimacy. Coordinates collapse without it.

**End Goal:** Pine-deployable state machine, not classroom theory.
