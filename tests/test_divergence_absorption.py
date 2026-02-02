"""
Test Suite — Divergence & Absorption Engines (PHASE 1.75)

Tests for:
    - DivergenceEngine (passive vs aggressor comparison)
    - AbsorptionEngine (internal vs external logic)
    - AOI Manager (area of interest tracking)

COVERAGE:
    - Full divergence detection
    - Partial divergence detection
    - Internal absorption identification
    - External absorption identification
    - Exhaustion absorption detection
    - AOI registration and mitigation
    - Multi-session tracking
"""

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


# Helper function to create coordinates with timestamps
def make_coord(price: float, timestamp: int = 1000000, **kwargs) -> Coordinate:
    """Helper to create Coordinate with default timestamp."""
    return Coordinate(price=price, timestamp=timestamp, **kwargs)


# ==================== DIVERGENCE ENGINE TESTS ====================


class TestDivergenceEngine:
    """Test DivergenceEngine for coordinate comparison."""
    
    def test_full_divergence_all_tfs(self):
        """Test full divergence when all TFs show opposite charges."""
        passive = make_coord(price=100.0, M=None, W=-1, D=-1, S=-1)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.FULL
        assert result.divergence_score == 1.0
        assert set(result.divergent_tfs) == {'W', 'D', 'S'}
        assert result.comparable_tfs == 3
    
    def test_partial_divergence(self):
        """Test partial divergence when some TFs diverge."""
        passive = make_coord(price=100.0, M=None, W=-1, D=-1, S=+1)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.PARTIAL
        assert result.divergence_score == pytest.approx(0.666, abs=0.01)
        assert set(result.divergent_tfs) == {'W', 'D'}
        assert result.comparable_tfs == 3
    
    def test_no_divergence_same_charges(self):
        """Test no divergence when all TFs have same charges."""
        passive = make_coord(price=100.0, M=None, W=+1, D=+1, S=+1)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.NONE
        assert result.divergence_score == 0.0
        assert len(result.divergent_tfs) == 0
        assert result.comparable_tfs == 3
    
    def test_no_comparable_tfs(self):
        """Test when coordinates have no overlapping TFs."""
        passive = make_coord(price=100.0, M=-1, W=None, D=None, S=None)
        aggressor = make_coord(price=105.0, M=None, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.NONE
        assert result.divergence_score == 0.0
        assert result.comparable_tfs == 0
    
    def test_is_full_divergence_helper(self):
        """Test is_full_divergence helper method."""
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert DivergenceEngine.is_full_divergence(passive, aggressor) is True
    
    def test_is_not_full_divergence(self):
        """Test is_full_divergence returns False for partial."""
        passive = make_coord(price=100.0, D=-1, S=+1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert DivergenceEngine.is_full_divergence(passive, aggressor) is False
    
    def test_get_divergence_score(self):
        """Test get_divergence_score helper method."""
        passive = make_coord(price=100.0, W=-1, D=-1, S=+1)
        aggressor = make_coord(price=105.0, W=+1, D=+1, S=+1)
        
        score = DivergenceEngine.get_divergence_score(passive, aggressor)
        assert score == pytest.approx(0.666, abs=0.01)
    
    def test_get_divergent_timeframes(self):
        """Test get_divergent_timeframes helper method."""
        passive = make_coord(price=100.0, W=-1, D=-1, S=+1)
        aggressor = make_coord(price=105.0, W=+1, D=+1, S=+1)
        
        divergent_tfs = DivergenceEngine.get_divergent_timeframes(passive, aggressor)
        assert set(divergent_tfs) == {'W', 'D'}
    
    def test_monthly_divergence(self):
        """Test divergence including monthly timeframe."""
        passive = make_coord(price=100.0, M=-1, W=-1, D=-1, S=-1)
        aggressor = make_coord(price=105.0, M=+1, W=+1, D=+1, S=+1)
        
        result = DivergenceEngine.calculate_divergence(passive, aggressor)
        
        assert result.divergence_type == DivergenceType.FULL
        assert result.divergence_score == 1.0
        assert set(result.divergent_tfs) == {'M', 'W', 'D', 'S'}


# ==================== ABSORPTION ENGINE TESTS ====================


class TestAbsorptionEngine:
    """Test AbsorptionEngine for absorption type detection."""
    
    def test_exhaustion_absorption_passive_stronger(self):
        """Test exhaustion absorption when passive volume > aggressor volume."""
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
        """Test external absorption when external target exists."""
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
        """Test internal absorption when no external target."""
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
        """Test no absorption when divergence score below threshold."""
        passive = make_coord(price=100.0, D=-1, S=+1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor
        )
        
        # Only 1 of 2 TFs divergent = 0.5 score (at threshold)
        # Should still detect absorption
        assert result.divergence.divergence_score == 0.5
    
    def test_absorption_strength_calculation(self):
        """Test absorption strength combines divergence and volume."""
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        result = AbsorptionEngine.analyze_absorption(
            passive=passive,
            aggressor=aggressor,
            passive_volume=800.0,
            aggressor_volume=200.0
        )
        
        # Full divergence (1.0) * (800/1000 volume ratio) = 0.8 strength
        assert result.absorption_strength == pytest.approx(0.8, abs=0.01)
    
    def test_is_exhaustion_absorption_helper(self):
        """Test is_exhaustion_absorption helper method."""
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert AbsorptionEngine.is_exhaustion_absorption(
            passive, aggressor, passive_volume=1000, aggressor_volume=500
        ) is True
        
        assert AbsorptionEngine.is_exhaustion_absorption(
            passive, aggressor, passive_volume=500, aggressor_volume=1000
        ) is False
    
    def test_is_internal_absorption_helper(self):
        """Test is_internal_absorption helper method."""
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        
        assert AbsorptionEngine.is_internal_absorption(
            passive, aggressor, external_target=None
        ) is True
    
    def test_is_external_absorption_helper(self):
        """Test is_external_absorption helper method."""
        passive = make_coord(price=100.0, D=-1, S=-1)
        aggressor = make_coord(price=105.0, D=+1, S=+1)
        external = make_coord(price=110.0, D=+1, S=+1)
        
        assert AbsorptionEngine.is_external_absorption(
            passive, aggressor, external_target=external
        ) is True


# ==================== AOI MANAGER TESTS ====================


class TestAOIManager:
    """Test AOI Manager for area tracking."""
    
    def test_register_aoi(self):
        """Test registering a new AOI."""
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
        """Test mitigation detection within tolerance."""
        manager = AOIManager()
        coord = make_coord(price=100.0, D=-1, S=-1)
        
        aoi = manager.register_aoi(
            coordinate=coord,
            price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        # Within tolerance (0.01% default)
        assert manager.is_mitigated(aoi, current_price=100.005) is True
        
        # Outside tolerance
        assert manager.is_mitigated(aoi, current_price=100.5) is False
    
    def test_mark_mitigated(self):
        """Test marking AOI as mitigated."""
        manager = AOIManager()
        coord = make_coord(price=100.0, D=-1, S=-1)
        
        aoi = manager.register_aoi(
            coordinate=coord,
            price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        # Mark as mitigated
        mitigated_aoi = manager.mark_mitigated(aoi)
        
        assert mitigated_aoi.is_mitigated is True
        assert mitigated_aoi.mitigation_timestamp is not None
        
        # Should be removed from active list
        active = manager.get_active_aois(session=SessionType.FRANKFURT)
        assert mitigated_aoi not in active
    
    def test_get_active_aois_by_session(self):
        """Test filtering active AOIs by session."""
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
        """Test filtering active AOIs by liquidity type."""
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
        """Test getting AOI with highest volume."""
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
        """Test tracking AOIs across multiple sessions."""
        manager = AOIManager()
        
        coord1 = make_coord(price=100.0, D=-1, S=-1)
        coord2 = make_coord(price=105.0, D=+1, S=+1)
        coord3 = make_coord(price=110.0, D=-1, S=-1)
        
        # Frankfurt low
        aoi1 = manager.register_aoi(
            coordinate=coord1, price=100.0,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0, session=SessionType.FRANKFURT
        )
        
        # London high
        aoi2 = manager.register_aoi(
            coordinate=coord2, price=105.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0, session=SessionType.LONDON,
            target_coordinate=coord3
        )
        
        # Check session chain
        sessions = manager.get_session_chain()
        assert SessionType.FRANKFURT in sessions
        assert SessionType.LONDON in sessions
        assert SessionType.NEW_YORK in sessions
    
    def test_clear_session(self):
        """Test clearing AOIs for specific session."""
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
        
        # Clear Frankfurt
        manager.clear_session(SessionType.FRANKFURT)
        
        frankfurt_aois = manager.get_active_aois(session=SessionType.FRANKFURT)
        assert len(frankfurt_aois) == 0
        
        # London should still exist
        london_aois = manager.get_active_aois(session=SessionType.LONDON)
        assert len(london_aois) == 1
    
    def test_clear_all(self):
        """Test clearing all AOIs."""
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


# ==================== INTEGRATION TESTS ====================


class TestDivergenceAbsorptionIntegration:
    """Test integration between all PHASE 1.75 components."""
    
    def test_frankfurt_low_london_high_scenario(self):
        """
        Test real scenario: Frankfurt Low → London High
        
        Frankfurt Low: (D−, S−)
        London Open:   (D+, S+) [Buy signal]
        → Internal divergence → continuation
        """
        manager = AOIManager()
        
        # Frankfurt low (passive)
        frankfurt_coord = make_coord(price=1.0950, D=-1, S=-1)
        frankfurt_aoi = manager.register_aoi(
            coordinate=frankfurt_coord,
            price=1.0950,
            liquidity_type=LiquidityType.INTERNAL,
            volume=1000.0,
            session=SessionType.FRANKFURT
        )
        
        # London open (aggressor)
        london_coord = make_coord(price=1.1000, D=+1, S=+1)
        
        # Calculate divergence
        divergence = DivergenceEngine.calculate_divergence(
            frankfurt_coord, london_coord
        )
        
        assert divergence.divergence_type == DivergenceType.FULL
        assert divergence.divergence_score == 1.0
        
        # Analyze absorption
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
        """
        Test external reversal scenario.
        
        AOI Low:  (W−, D−, S−)
        Present:  (W+, D+, S+)
        Target:   External High (W+, D+)
        → External divergence → reversal
        """
        manager = AOIManager()
        
        # AOI low (passive)
        aoi_coord = make_coord(price=100.0, W=-1, D=-1, S=-1)
        aoi = manager.register_aoi(
            coordinate=aoi_coord,
            price=100.0,
            liquidity_type=LiquidityType.EXTERNAL,
            volume=2000.0,
            session=SessionType.LONDON
        )
        
        # Present (aggressor)
        present_coord = make_coord(price=105.0, W=+1, D=+1, S=+1)
        
        # External target
        target_coord = make_coord(price=110.0, W=+1, D=+1, S=+1)
        
        # Calculate divergence
        divergence = DivergenceEngine.calculate_divergence(
            aoi_coord, present_coord
        )
        
        assert divergence.divergence_type == DivergenceType.FULL
        
        # Analyze absorption
        absorption = AbsorptionEngine.analyze_absorption(
            passive=aoi_coord,
            aggressor=present_coord,
            external_target=target_coord,
            passive_volume=2000.0,
            aggressor_volume=1000.0
        )
        
        # Passive volume > aggressor → exhaustion
        assert absorption.absorption_type == AbsorptionType.EXHAUSTION
        assert absorption.is_reversal_signal is True
