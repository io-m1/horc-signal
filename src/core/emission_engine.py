from dataclasses import dataclass
from typing import Optional
from enum import Enum

class AbsorptionType(Enum):
    NONE = "none"
    INTERNAL = "internal"      # Aligned energy → continuation
    EXTERNAL = "external"      # Opposing energy → rejection
    EXHAUSTION = "exhaustion"  # High effort, low result → stall

@dataclass(frozen=True)
class EmissionResult:
    emission: float
    emission_norm: float
    displacement: float
    volume: float
    absorption_type: AbsorptionType
    is_near_defended: bool

@dataclass(frozen=True)
class EmissionDivergenceResult:
    emission_divergence: bool
    price_divergence: bool
    price_velocity: float
    emission_3bar: float
    divergence_axes: int
    is_full_divergence: bool
    is_partial_divergence: bool

class EmissionEngine:
    EXHAUSTION_THRESHOLD = 1.5  # High effort, minimal result
    INTERNAL_THRESHOLD = 1.2    # Aligned energy
    EXTERNAL_THRESHOLD = 1.0    # Opposing energy
    
    EMISSION_DIV_THRESHOLD = 1.4  # High emission floor
    EMISSION_VEL_CAP = 0.3        # Low velocity cap
    PRICE_DIV_THRESHOLD = 0.6     # High velocity floor
    PRICE_EMISS_CAP = 0.8         # Low emission cap
    
    DEFENDED_LIQ_PROXIMITY = 0.4  # ATR multiplier
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self._emission_history: list[float] = []
    
    def calculate_emission(
        self,
        close: float,
        open_price: float,
        volume: float,
        atr: float,
        defended_liq: Optional[float] = None,
        intent_balance: float = 0.0,
        current_participant: int = 0,
        close_prev: float = None
    ) -> EmissionResult:
        body = abs(close - open_price)
        displacement = max(body, atr * 0.1)
        
        emission = volume / displacement if displacement > 0 else 0.0
        
        self._emission_history.append(emission)
        if len(self._emission_history) > self.lookback:
            self._emission_history.pop(0)
        
        emission_sma = sum(self._emission_history) / len(self._emission_history) if self._emission_history else 1.0
        emission_norm = emission / emission_sma if emission_sma > 0 else 0.0
        
        is_near_defended = False
        if defended_liq is not None:
            is_near_defended = abs(close - defended_liq) < atr * self.DEFENDED_LIQ_PROXIMITY
        
        absorption_type = AbsorptionType.NONE
        
        if is_near_defended:
            if close_prev is not None and emission_norm > self.EXHAUSTION_THRESHOLD:
                if abs(close - close_prev) < atr * 0.25:
                    absorption_type = AbsorptionType.EXHAUSTION
            
            if absorption_type == AbsorptionType.NONE and emission_norm > self.INTERNAL_THRESHOLD:
                if intent_balance * current_participant > 0:
                    absorption_type = AbsorptionType.INTERNAL
            
            if absorption_type == AbsorptionType.NONE and emission_norm > self.EXTERNAL_THRESHOLD:
                absorption_type = AbsorptionType.EXTERNAL
        
        return EmissionResult(
            emission=emission,
            emission_norm=emission_norm,
            displacement=displacement,
            volume=volume,
            absorption_type=absorption_type,
            is_near_defended=is_near_defended
        )
    
    def calculate_divergence(
        self,
        close: float,
        close_3bars_ago: float,
        atr: float,
        emission_current: float,
        emission_1bar: float,
        emission_2bar: float,
        expected_dir: int = 0,
        intent_balance: float = 0.0
    ) -> EmissionDivergenceResult:
        emission_3bar = (emission_current + emission_1bar + emission_2bar) / 3.0
        
        price_vel = (close - close_3bars_ago) / (atr * 3.0) if atr > 0 else 0.0
        
        emission_div = emission_3bar > self.EMISSION_DIV_THRESHOLD and abs(price_vel) < self.EMISSION_VEL_CAP
        
        price_div = abs(price_vel) > self.PRICE_DIV_THRESHOLD and emission_3bar < self.PRICE_EMISS_CAP
        
        div_axes = 0
        if emission_div:
            div_axes += 1
        if price_div:
            div_axes += 1
        
        if div_axes > 0 and expected_dir != 0:
            if intent_balance * expected_dir < 0:
                div_axes += 1
        
        return EmissionDivergenceResult(
            emission_divergence=emission_div,
            price_divergence=price_div,
            price_velocity=price_vel,
            emission_3bar=emission_3bar,
            divergence_axes=div_axes,
            is_full_divergence=div_axes >= 2,
            is_partial_divergence=div_axes == 1
        )
    
    def reset(self):
        self._emission_history.clear()
