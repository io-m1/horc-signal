import pytest
from datetime import datetime

from src.core.coordinate_engine import Coordinate
from src.core.divergence_engine import (
    DivergenceEngine,
    DivergenceType,
    DivergenceResult
)
from src.core.absorption_engine import (
    AbsorptionEngine,
    AbsorptionType,
    AbsorptionResult
)
from src.core.aoi_manager import (
    AOIManager,
    AOI,
    LiquidityType,
    SessionType
)

def make_coord(price: float, timestamp: int = 1000000, **kwargs) -> Coordinate:
    return Coordinate(price=price, timestamp=timestamp, **kwargs)

class TestDivergenceEngine:
    
    def test_full_divergence_all_tfs(self):
        passive = make_coord(price=100.0, M=None, W=-1, D=-1, S=-1)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.FULL
        assert result.divergence_score == 1.0
        assert set(result.divergent_tfs) == {'W', 'D', 'S'}
        assert result.comparable_tfs == 3
    
    def test_partial_divergence(self):
        passive = make_coord(price=100.0, M=None, W=-1, D=-1, S=+1)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.PARTIAL
        assert result.divergence_score == pytest.approx(0.666, abs=0.01)
        assert set(result.divergent_tfs) == {'W', 'D'}
        assert result.comparable_tfs == 3
    
    def test_no_divergence_same_charges(self):
        passive = make_coord(price=100.0, M=None, W=+1, D=+1, S=+1)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.NONE
        assert result.divergence_score == 0.0
        assert len(result.divergent_tfs) == 0
        assert result.comparable_tfs == 3
    
    def test_no_comparable_tfs(self):
        passive = make_coord(price=100.0, M=-1, W=None, D=None, S=None)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.NONE
        assert result.divergence_score == 0.0
        assert result.comparable_tfs == 0
    
    def test_is_full_divergence_helper(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert DivergenceEngine.is_full_divergence(passive, aggressor) is True
    
    def test_is_not_full_divergence(self):
        passive = make_coord(price=100.0, D=-1, S=+1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert DivergenceEngine.is_full_divergence(passive, aggressor) is False
    
    def test_get_divergence_score(self):
        passive = make_coord(price=100.0, W=-1, D=-1, S=+1)
        aggressor = make_coord(price=105.0, W=+1, D=+1, S=+1)
        
        score = DivergenceEngine.get_divergence_score(passive, aggressor)
        assert score == pytest.approx(0.666, abs=0.01)
    
    def test_get_divergent_timeframes(self):
        passive = make_coord(price=100.0, W=-1, D=-1, S=+1)
        aggressor = make_coord(price=105.0, W=+1, D=+1, S=+1)
        
        divergent_tfs = DivergenceEngine.get_divergent_timeframes(passive, aggressor)
        assert set(divergent_tfs) == {'W', 'D'}
    
    def test_monthly_divergence(self):
        passive = make_coord(price=100.0, M=-1, W=-1, D=-1, S=-1)
        aggressor = make_coord(price=105.0, M=+1, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.FULL
        assert result.divergence_score == 1.0
        assert set(result.divergent_tfs) == {'M', 'W', 'D', 'S'}

class TestAbsorptionEngine:
    
    def test_exhaustion_absorption_passive_stronger(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            passive_volume=1000.0,
            aggressor_volume=500.0
        )
        
        assert result.absorption_type == AbsorptionType.EXHAUSTION
        assert result.is_reversal_signal is True
        assert result.is_continuation_signal is False
    
    def test_external_absorption_with_target(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        external = make_coord(price=110.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            external_target=external,
            passive_volume=500.0,
            aggressor_volume=1000.0
        )
        
        assert result.absorption_type == AbsorptionType.EXTERNAL
        assert result.is_reversal_signal is True
        assert result.target_coordinate == external
    
    def test_internal_absorption_no_external(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            external_target=None,
            passive_volume=500.0,
            aggressor_volume=1000.0
        )
        
        assert result.absorption_type == AbsorptionType.INTERNAL
        assert result.is_continuation_signal is True
        assert result.is_reversal_signal is False
    
    def test_no_absorption_below_threshold(self):
        passive = make_coord(price=100.0, D=-1, S=+1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor
        )
        
        assert result.divergence.divergence_score == 0.5
    
    def test_absorption_strength_calculation(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            passive_volume=800.0,
            aggressor_volume=200.0
        )
        
        assert result.absorption_strength == pytest.approx(0.8, abs=0.01)
    
    def test_is_exhaustion_absorption_helper(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert AbsorptionEngine.is_exhaustion_absorption(
            passive, aggressor, passive_volume=1000, aggressor_volume=500
        ) is True
        
        assert AbsorptionEngine.is_exhaustion_absorption(
            passive, aggressor, passive_volume=500, aggressor_volume=1000
        ) is False
    
    def test_is_internal_absorption_helper(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert AbsorptionEngine.is_internal_absorption(
            passive, aggressor, external_target=None
        ) is True
    
    def test_is_external_absorption_helper(self):
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        external = make_coord(price=110.0, D=+1, S=+1)
        
        assert AbsorptionEngine.is_external_absorption(
            passive, aggressor, external_target=external
        ) is True

class TestAOIManager:
    
    def test_register_aoi(self):
        manager = AOIManager()
        coord = make_coord(price=100.0, D=-1, S=-1)
        
        aoi = manager.register_aoi(
            coordinate=coord,
            price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        assert aoi.price == 100.0
        assert aoi.coordinate == coord
        assert aoi.liquidity_type == LiquidityType.INTERNAL
        assert aoi.volume == 1000.0
        assert aoi.session == SessionType.FRANKFURT
        assert aoi.is_mitigated is False
    
    def test_is_mitigated_within_tolerance(self):
        manager = AOIManager()
        coord = make_coord(price=100.0, D=-1, S=-1)
        
        aoi = manager.register_aoi(
            coordinate=coord,
            price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        assert manager.is_mitigated(aoi, current_price=100.005) is True
        
        assert manager.is_mitigated(aoi, current_price=100.5) is False
    
    def test_mark_mitigated(self):
        manager = AOIManager()
        coord = make_coord(price=100.0, D=-1, S=-1)
        
        aoi = manager.register_aoi(
            coordinate=coord,
            price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        mitigated_aoi = manager.mark_mitigated(aoi)
        
        assert mitigated_aoi.is_mitigated is True
        assert mitigated_aoi.mitigation_timestamp is not None
        
        active = manager.get_active_aois(session=SessionType.FRANKFURT)
        assert mitigated_aoi not in active
    
    def test_get_active_aois_by_session(self):
        manager = AOIManager()
        
        coord1 = make_coord(price=100.0, D=-1, S=-1)
        coord2 = make_coord(price=105.0, D=+1, S=+1)
        
        aoi1 = manager.register_aoi(
            coordinate=coord1, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        aoi2 = manager.register_aoi(
            coordinate=coord2, price=105.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0, session=SessionType.LONDON
        )
        
        frankfurt_aois = manager.get_active_aois(session=SessionType.FRANKFURT)
        assert len(frankfurt_aois) == 1
        assert frankfurt_aois[0] == aoi1
        
        london_aois = manager.get_active_aois(session=SessionType.LONDON)
        assert len(london_aois) == 1
        assert london_aois[0] == aoi2
    
    def test_get_active_aois_by_liquidity_type(self):
        manager = AOIManager()
        
        coord1 = make_coord(price=100.0, D=-1, S=-1)
        coord2 = make_coord(price=105.0, D=+1, S=+1)
        
        aoi1 = manager.register_aoi(
            coordinate=coord1, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        aoi2 = manager.register_aoi(
            coordinate=coord2, price=105.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0, session=SessionType.FRANKFURT
        )
        
        internal_aois = manager.get_active_aois(
            liquidity_type=LiquidityType.INTERNAL
        )
        assert len(internal_aois) == 1
        assert internal_aois[0] == aoi1
    
    def test_get_highest_volume_aoi(self):
        manager = AOIManager()
        
        coord1 = make_coord(price=100.0, D=-1, S=-1)
        coord2 = make_coord(price=105.0, D=+1, S=+1)
        coord3 = make_coord(price=110.0, D=-1, S=-1)
        
        aoi1 = manager.register_aoi(
            coordinate=coord1, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        aoi2 = manager.register_aoi(
            coordinate=coord2, price=105.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=3000.0, session=SessionType.FRANKFURT
        )
        
        aoi3 = manager.register_aoi(
            coordinate=coord3, price=110.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=2000.0, session=SessionType.FRANKFURT
        )
        
        highest = manager.get_highest_volume_aoi(session=SessionType.FRANKFURT)
        assert highest == aoi2
        assert highest.volume == 3000.0
    
    def test_multi_session_tracking(self):
        manager = AOIManager()
        
        coord1 = make_coord(price=100.0, D=-1, S=-1)
        coord2 = make_coord(price=105.0, D=+1, S=+1)
        coord3 = make_coord(price=110.0, D=-1, S=-1)
        
        aoi1 = manager.register_aoi(
            coordinate=coord1, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        aoi2 = manager.register_aoi(
            coordinate=coord2, price=105.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0, session=SessionType.LONDON,
            target_coordinate=coord3
        )
        
        sessions = manager.get_session_chain()
        assert SessionType.FRANKFURT in sessions
        assert SessionType.LONDON in sessions
        assert SessionType.NEW_YORK in sessions
    
    def test_clear_session(self):
        manager = AOIManager()
        
        coord1 = make_coord(price=100.0, D=-1, S=-1)
        coord2 = make_coord(price=105.0, D=+1, S=+1)
        
        manager.register_aoi(
            coordinate=coord1, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        manager.register_aoi(
            coordinate=coord2, price=105.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0, session=SessionType.LONDON
        )
        
        manager.clear_session(SessionType.FRANKFURT)
        
        frankfurt_aois = manager.get_active_aois(session=SessionType.FRANKFURT)
        assert len(frankfurt_aois) == 0
        
        london_aois = manager.get_active_aois(session=SessionType.LONDON)
        assert len(london_aois) == 1
    
    def test_clear_all(self):
        manager = AOIManager()
        
        coord = make_coord(price=100.0, D=-1, S=-1)
        
        manager.register_aoi(
            coordinate=coord, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        manager.clear_all()
        
        all_aois = manager.get_all_aois()
        assert len(all_aois) == 0

class TestDivergenceAbsorptionIntegration:
    
    def test_frankfurt_low_london_high_scenario(self):
        manager = AOIManager()
        
        frankfurt_coord = make_coord(price=1.0950, D=-1, S=-1)
        frankfurt_aoi = manager.register_aoi(
            coordinate=frankfurt_coord,
            price=1.0950,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        london_coord = make_coord(price=1.1000, D=+1, S=+1)
        
        divergence = DivergenceEngine.calculate_divergence(
            frankfurt_coord, london_coord
        )
        
        assert divergence.divergence_type == DivergenceType.FULL
        assert divergence.divergence_score == 1.0
        
        absorption = AbsorptionEngine.analyze_absorption(
            passive=frankfurt_coord,
            aggressor=london_coord,
            external_target=None,
            passive_volume=1000.0,
            aggressor_volume=1500.0
        )
        
        assert absorption.absorption_type == AbsorptionType.INTERNAL
        assert absorption.is_continuation_signal is True
    
    def test_external_reversal_scenario(self):
        manager = AOIManager()
        
        aoi_coord = make_coord(price=100.0, W=-1, D=-1, S=-1)
        aoi = manager.register_aoi(
            coordinate=aoi_coord,
            price=100.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0,
            session=SessionType.LONDON
        )
        
        present_coord = make_coord(price=105.0, W=+1, D=+1, S=+1)
        
        target_coord = make_coord(price=110.0, W=+1, D=+1, S=+1)
        
        divergence = DivergenceEngine.calculate_divergence(
            aoi_coord, present_coord
        )
        
        assert divergence.divergence_type == DivergenceType.FULL
        
        absorption = AbsorptionEngine.analyze_absorption(
            passive=aoi_coord,
            aggressor=present_coord,
            external_target=target_coord,
            passive_volume=2000.0,
            aggressor_volume=1000.0
        )
        
        assert absorption.absorption_type == AbsorptionType.EXHAUSTION
        assert absorption.is_reversal_signal is True
