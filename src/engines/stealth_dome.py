from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

# REMOVED EXTERNAL IMPORT OF MTFResampler
from src.engines.horc_coordinates import (
    CoordinateTracker, RangeAnalysis, RangeType, Participant
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Enums
# ─────────────────────────────────────────────────────────────────────────────

BUYER = 1
SELLER = -1
NEUTRAL = 0

TF_DAILY_TIER = ["D", "720", "480", "360", "240"]
TF_SESSION_TIER = ["180", "120", "60"]


@dataclass
class OHLCV:
    """Standard candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    bar_count: int = 1
    start_time: Optional[datetime] = None

    def __post_init__(self):
        # Ensure we don't crash if volume is None or other issues
        if self.volume is None:
            self.volume = 0.0

    def update(self, other: OHLCV):
        """Merge 'other' (newer) into 'self' (running candle)."""
        self.high = max(self.high, other.high)
        self.low = min(self.low, other.low)
        self.close = other.close
        self.volume += other.volume
        self.bar_count += other.bar_count
        # start_time remains self.start_time

    def reset(self):
        self.timestamp = datetime.min
        self.open = 0.0
        self.high = -math.inf
        self.low = math.inf
        self.close = 0.0
        self.volume = 0.0
        self.bar_count = 0
        self.start_time = None


@dataclass
class StealthSignal:
    """Output of the Stealth Dome engine on a signal bar."""
    timestamp: datetime
    direction: int         # BUYER (+1) or SELLER (-1)
    entry_price: float     # close of signal bar
    sl_price: float        # stop-loss
    tp_price: float        # take-profit (2R default)
    bias_tf: str           # which TF drove the bias
    divergence_score: int  # how many TFs disagree
    max_divergence: int    # total TFs checked
    crt_type: str          # "CRT_LONG" or "CRT_SHORT"
    kill_zone: str         # "LONDON" / "NY" / "ASIA" / "OFF"
    prev_high: float       # CRT reference candle high
    prev_low: float        # CRT reference candle low


@dataclass
class StealthDomeConfig:
    """Mirrors PineScript inputs."""
    ltf_minutes: int = 5          # First Raid Pulse TF
    crt_active: bool = True       # Active CRT Filter
    crt_tf_minutes: int = 0       # CRT Candle TF (0 = chart TF)
    box_look_ahead: int = 10      # Box Look-Ahead Bars
    min_divergence: int = 2       # Min Divergence Score
    atr_sl_buffer: float = 0.3    # ATR SL Buffer (v8.9b Beginner Friendly)
    use_kill_zones: bool = True   # Filter by Kill Zones
    risk_reward: float = 1.5      # R:R for TP calculation (Calibration Golden Set)
    chart_tf_minutes: int = 5     # chart timeframe for CRT when crt_tf_minutes==0
    # v8.7 Configs
    horc_ref_offset: int = 0      # Candle lookback for HORC/CRT reference.
                                  # 0 = previous closed candle (Pine high[1]).
                                  # 6 = Pine high[7]. Default 0 per audit.
    divergence_mode: str = "conflict"  # "conflict": signal valid when TFs disagree
                                       # "consensus": signal valid when TFs agree
    # v8.8 Configs
    use_coordinates_filter: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe resampler
# ─────────────────────────────────────────────────────────────────────────────

def _tf_minutes(tf_label: str) -> int:
    """Convert label to minutes. 'D' = 1440."""
    if tf_label.upper() == "D":
        return 1440
    return int(tf_label)


def _is_new_period(current_ts: datetime, prev_ts: Optional[datetime], period_min: int) -> bool:
    """Check if we rolled into a new TF period (floor-based)."""
    if prev_ts is None:
        return True
    if period_min >= 1440:
        return current_ts.date() != prev_ts.date()
    total_cur = current_ts.hour * 60 + current_ts.minute
    total_prev = prev_ts.hour * 60 + prev_ts.minute
    bucket_cur = total_cur // period_min
    bucket_prev = total_prev // period_min
    return bucket_cur != bucket_prev or current_ts.date() != prev_ts.date()


# How many completed candles to keep per TF (need [7] lookback = 8 slots)
_HISTORY_DEPTH = 8

@dataclass
class MTFCandle:
    """Accumulator for building a higher-timeframe candle."""
    timestamp: datetime = field(default_factory=lambda: datetime.min)
    open: float = 0.0
    high: float = -math.inf
    low: float = math.inf
    close: float = 0.0
    volume: float = 0.0
    bar_count: int = 0
    start_time: Optional[datetime] = None

    def update(self, bar: OHLCV):
        if self.bar_count == 0:
            self.timestamp = bar.timestamp  # This will be the closing timestamp of the bucket usually?
                                            # Actually we usually align timestamp to period end or start.
                                            # Here we just take the first bar's timestamp or keep updating.
                                            # The resampler logic handles the specific timestamp management.
            self.open = bar.open
            self.high = bar.high
            self.low = bar.low
            self.close = bar.close
            self.volume = bar.volume
            self.start_time = bar.timestamp
        else:
            self.high = max(self.high, bar.high)
            self.low = min(self.low, bar.low)
            self.close = bar.close
            self.volume += bar.volume
        
        self.bar_count += 1
        # Always update timestamp to current bar (so it represents 'closing time' roughly)
        self.timestamp = bar.timestamp

    def to_ohlcv(self) -> OHLCV:
        return OHLCV(
            timestamp=self.timestamp,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            bar_count=self.bar_count,
            start_time=self.start_time
        )


class MTFResampler:
    """Maintains rolling higher-TF candles from 1-min input bars.

    v8.6 change: stores a ring buffer of _HISTORY_DEPTH completed candles
    per TF, enabling high[7] / low[7] Closing Range Logic lookbacks.
    """

    def __init__(self, tf_labels: List[str]):
        self.tf_labels = tf_labels
        self._periods = {tf: _tf_minutes(tf) for tf in tf_labels}
        self._current: Dict[str, MTFCandle] = {tf: MTFCandle() for tf in tf_labels}
        # Ring buffer of completed candles per TF (index 0 = most recent)
        self._history: Dict[str, deque] = {
            tf: deque(maxlen=_HISTORY_DEPTH) for tf in tf_labels
        }
        self._last_ts: Dict[str, Optional[datetime]] = {tf: None for tf in tf_labels}

    def update(self, bar: OHLCV):
        """Feed one 1-min bar. Closes any completed periods."""
        for tf in self.tf_labels:
            period = self._periods[tf]
            if _is_new_period(bar.timestamp, self._last_ts[tf], period):
                # Close previous candle into history buffer
                if self._current[tf].bar_count > 0:
                    self._history[tf].appendleft(self._current[tf].to_ohlcv())
                self._current[tf] = MTFCandle()
            self._current[tf].update(bar)
            self._last_ts[tf] = bar.timestamp

    def on_new_bar(self, bar: OHLCV) -> Dict[str, OHLCV]:
        """
        Processes a new 1-minute bar and returns a dictionary of
        completed OHLCV candles for each TF that just closed.
        """
        closed_candles: Dict[str, OHLCV] = {}
        for tf in self.tf_labels:
            period = self._periods[tf]
            if _is_new_period(bar.timestamp, self._last_ts[tf], period):
                # Close previous candle into history buffer
                if self._current[tf].bar_count > 0:
                    completed_candle = self._current[tf].to_ohlcv()
                    self._history[tf].appendleft(completed_candle)
                    closed_candles[tf] = completed_candle
                self._current[tf] = MTFCandle()
            self._current[tf].update(bar)
            self._last_ts[tf] = bar.timestamp
        return closed_candles

    def get_candle_ago(self, tf: str, n: int) -> Optional[OHLCV]:
        """Return the completed candle N bars ago on this TF.
        n=0 -> most recently completed, n=1 -> the one before, ... n=7 -> high[7].
        Maps to PineScript's request.security(tf, high[n]).
        """
        hist = self._history.get(tf)
        if hist is None or len(hist) <= n:
            return None
        return hist[n]

    def get_prev_candle(self, tf: str) -> Optional[OHLCV]:
        """Equivalent of request.security(tf, high[1]) — returns most recently completed candle."""
        return self.get_candle_ago(tf, 0)

    def get_current_candle(self, tf: str) -> Optional[OHLCV]:
        """Returns the candle currently being built for TF."""
        c = self._current.get(tf)
        if c and c.bar_count > 0:
            return c.to_ohlcv()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ATR calculator
# ─────────────────────────────────────────────────────────────────────────────

class ATR:
    """Rolling ATR(14) on the chart-TF bars."""

    def __init__(self, period: int = 14):
        self.period = period
        self._trs: deque = deque(maxlen=period)
        self._prev_close: Optional[float] = None

    def update(self, bar: OHLCV) -> float:
        hl = bar.high - bar.low
        if self._prev_close is not None:
            hc = abs(bar.high - self._prev_close)
            lc = abs(bar.low - self._prev_close)
            tr = max(hl, hc, lc)
        else:
            tr = hl
        self._trs.append(tr)
        self._prev_close = bar.close
        if len(self._trs) == 0:
            return 0.0
        return sum(self._trs) / len(self._trs)


# ─────────────────────────────────────────────────────────────────────────────
# Stealth Dome Engine
# ─────────────────────────────────────────────────────────────────────────────

class StealthDomeEngine:
    """
    Full Python port of the PineScript v6 HORC-CRT Stealth Dome indicator.

    Feed it 1-minute bars chronologically; it emits StealthSignal when
    all conditions align: HORC bias, divergence score, CRT pattern, kill zone.
    """

    def __init__(self, config: Optional[StealthDomeConfig] = None):
        self.cfg = config or StealthDomeConfig()

        # MTF resamplers
        all_tfs = TF_DAILY_TIER + TF_SESSION_TIER
        self.resampler = MTFResampler(all_tfs)

        # LTF resampler (5-min by default) for raid detection
        self.ltf_resampler = MTFResampler([str(self.cfg.ltf_minutes)])
        self.ltf_label = str(self.cfg.ltf_minutes)

        # CRT resampler
        crt_min = self.cfg.crt_tf_minutes if self.cfg.crt_tf_minutes > 0 else self.cfg.chart_tf_minutes
        self.crt_tf_label = str(crt_min)
        self.crt_resampler = MTFResampler([self.crt_tf_label])

        # HORC tf_history map: tf → last raid direction
        self.tf_history: Dict[str, int] = {tf: NEUTRAL for tf in all_tfs}

        # ATR on CRT-TF bars
        self.atr = ATR(14)
        self._atr_val: float = 0.0

        # Chart-TF resampler for ATR computation (matches CRT TF)
        self._chart_resampler = MTFResampler([self.crt_tf_label])
        self._last_chart_candle: Optional[OHLCV] = None
        
        # v8.8 Coordinate Tracker
        self.coord_tracker = CoordinateTracker()

    # ── PineScript f_scan ────────────────────────────────────────────────
    def _scan_raid(self, ref_high: float, ref_low: float) -> int:
        """Check if the LTF (5-min) bar swept above refH or below refL."""
        ltf_candle = self.ltf_resampler.get_current_candle(self.ltf_label)
        if ltf_candle is None:
            return NEUTRAL
        if ltf_candle.high > ref_high:
            return BUYER
        if ltf_candle.low < ref_low:
            return SELLER
        return NEUTRAL

    # ── PineScript f_horc ────────────────────────────────────────────────
    def _horc_tf(self, tf: str, is_new_period: bool) -> int:
        """Detect conviction flip per TF."""
        ref_candle = self.resampler.get_candle_ago(tf, self.cfg.horc_ref_offset)
        if ref_candle is None:
            return NEUTRAL

        raid = self._scan_raid(ref_candle.high, ref_candle.low)
        prev = self.tf_history[tf]
        con = (raid != prev) and (raid != NEUTRAL)
        sig = raid if con else NEUTRAL

        if is_new_period:
            self.tf_history[tf] = raid

        return sig

    # ── PineScript f_get_tier_signal ─────────────────────────────────────
    def _get_tier_signal(self, tfs: List[str], new_period_map: Dict[str, bool]) -> Tuple[int, str]:
        """Cascade priority scan across TF tier."""
        for tf in tfs:
            sig = self._horc_tf(tf, new_period_map.get(tf, False))
            if sig != NEUTRAL:
                # Check v8.8 Coordinate Filter
                if self.cfg.use_coordinates_filter:
                    # Analyze Range Context
                    current_price = self._last_chart_candle.close if self._last_chart_candle else 0.0
                    if current_price == 0.0:
                        continue

                    ctx = RangeAnalysis.analyze(self.coord_tracker, tf, current_price)
                    
                    is_valid_zone = True
                    if sig == BUYER: # BUY Signal
                        if ctx.range_type == RangeType.PREMIUM:
                            is_valid_zone = False # Don't buy in Premium
                    elif sig == SELLER: # SELL Signal
                        if ctx.range_type == RangeType.DISCOUNT:
                            is_valid_zone = False # Don't sell in Discount
                    
                    if not is_valid_zone:
                        continue # Skip this TF signal
            
                return sig, tf
        return NEUTRAL, "N/A"

    # ── PineScript f_calc_div ────────────────────────────────────────────
    def _calc_divergence(self, bias: int) -> int:
        """Count how many TFs disagree with bias."""
        if bias == NEUTRAL:
            return 0
        div = 0
        for tf in TF_DAILY_TIER + TF_SESSION_TIER:
            s = self.tf_history[tf]
            if s != NEUTRAL and s != bias:
                div += 1
        return div

    # ── Kill zone check ──────────────────────────────────────────────────
    @staticmethod
    def _in_kill_zone(ts: datetime) -> Tuple[bool, str]:
        """Check if timestamp falls in London, NY, or Asia kill zone (UTC hours)."""
        h, m = ts.hour, ts.minute
        t = h * 60 + m
        if 7 * 60 <= t < 10 * 60:
            return True, "LONDON"
        if 12 * 60 <= t < 15 * 60:
            return True, "NY"
        if t >= 1 * 60 and t < 5 * 60:
            return True, "ASIA"
        return False, "OFF"

    # ── CRT check ────────────────────────────────────────────────────────
    def _check_crt(self, bar: OHLCV) -> Tuple[bool, bool]:
        """
        Returns (crt_long, crt_short).
        CRT Long:  bar sweeps below ref low, closes back above it.
        CRT Short: bar sweeps above ref high, closes back below it.

        v8.7: Uses configurable offset (default=0 = previous closed candle).
        """
        ref = self.crt_resampler.get_candle_ago(self.crt_tf_label, self.cfg.horc_ref_offset)
        if ref is None:
            return False, False

        crt_long = (bar.low < ref.low) and (bar.close > ref.low)
        crt_short = (bar.high > ref.high) and (bar.close < ref.high)
        return crt_long, crt_short

    # ── Main process bar ─────────────────────────────────────────────────
    def process_bar(self, bar: OHLCV) -> Optional[StealthSignal]:
        """
        Feed one 1-minute bar. Updates all internal state and returns a
        StealthSignal if all conditions align, else None.
        """
        # 1) Detect new periods BEFORE updating resamplers
        new_period_map: Dict[str, bool] = {}
        for tf in TF_DAILY_TIER + TF_SESSION_TIER:
            period = _tf_minutes(tf)
            prev_ts = self.resampler._last_ts.get(tf)
            new_period_map[tf] = _is_new_period(bar.timestamp, prev_ts, period)

        # 2) Update all resamplers and feed Coordinate Tracker
        closed_bars = self.resampler.on_new_bar(bar)
        
        # Feed Coordinate Tracker with closed bars
        for tf, closed_bar in closed_bars.items():
            self.coord_tracker.on_bar(
                timeframe=tf,
                time=closed_bar.timestamp, # Use timestamp from OHLCV
                open_=closed_bar.open,
                high=closed_bar.high,
                low=closed_bar.low,
                close=closed_bar.close,
                is_new_period=True
            )

        self.ltf_resampler.update(bar)
        self.crt_resampler.update(bar)

        # 3) Update ATR on CRT-TF candle completion
        crt_period = _tf_minutes(self.crt_tf_label)
        crt_prev_ts = self._chart_resampler._last_ts.get(self.crt_tf_label)
        if _is_new_period(bar.timestamp, crt_prev_ts, crt_period):
            completed = self._chart_resampler._current[self.crt_tf_label]
            if completed.bar_count > 0:
                self._atr_val = self.atr.update(completed.to_ohlcv())
        self._chart_resampler.update(bar)
        self._last_chart_candle = bar # v8.8 FIX: track current bar for range analysis

        # 4) HORC compass
        d_sig, d_tf = self._get_tier_signal(TF_DAILY_TIER, new_period_map)
        s_sig, s_tf = self._get_tier_signal(TF_SESSION_TIER, new_period_map)

        horc_bias = NEUTRAL
        active_tf = "N/A"
        if d_sig != NEUTRAL:
            horc_bias = d_sig
            active_tf = d_tf
        elif s_sig != NEUTRAL:
            horc_bias = s_sig
            active_tf = s_tf

        # 5) Divergence
        horc_div = self._calc_divergence(horc_bias)
        max_div = len(TF_DAILY_TIER) + len(TF_SESSION_TIER)
        # v8.7 FIX (Audit Issue #2 — divergence philosophy):
        # "conflict" mode: more TF disagreement required → absorption/flip signal
        # "consensus" mode: fewer TF disagreement required → trend strength signal
        if self.cfg.divergence_mode == "consensus":
            bias_valid = horc_div <= self.cfg.min_divergence
        else:  # "conflict" (original v8.6 behavior)
            bias_valid = horc_div >= self.cfg.min_divergence

        # 6) CRT
        crt_long, crt_short = self._check_crt(bar)

        # 7) Kill zone
        in_kz, kz_name = self._in_kill_zone(bar.timestamp)
        if self.cfg.use_kill_zones and not in_kz:
            return None

        # 8) Signal gating
        valid_long = bias_valid and horc_bias == BUYER and (not self.cfg.crt_active or crt_long)
        valid_short = bias_valid and horc_bias == SELLER and (not self.cfg.crt_active or crt_short)

        if not valid_long and not valid_short:
            return None

        # 9) Build signal
        direction = BUYER if valid_long else SELLER
        atr_val = self._atr_val if self._atr_val > 0 else (bar.high - bar.low)

        if direction == BUYER:
            sl = bar.low - atr_val * self.cfg.atr_sl_buffer
            risk = bar.close - sl
            tp = bar.close + risk * self.cfg.risk_reward
        else:
            sl = bar.high + atr_val * self.cfg.atr_sl_buffer
            risk = sl - bar.close
            tp = bar.close - risk * self.cfg.risk_reward

        prev_crt = self.crt_resampler.get_prev_candle(self.crt_tf_label)

        return StealthSignal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=bar.close,
            sl_price=sl,
            tp_price=tp,
            bias_tf=active_tf,
            divergence_score=horc_div,
            max_divergence=max_div,
            crt_type="CRT_LONG" if valid_long else "CRT_SHORT",
            kill_zone=kz_name if in_kz else "OFF",
            prev_high=prev_crt.high if prev_crt else 0.0,
            prev_low=prev_crt.low if prev_crt else 0.0,
        )
