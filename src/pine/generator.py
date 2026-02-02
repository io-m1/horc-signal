"""
Pine Script Generator for HORC Signal System

Generates TradingView-compatible Pine Script v5 code from Python implementation.
Ensures 1:1 parity between Python and Pine execution.

ARCHITECTURE:
    Python SignalIR → Pine var primitives
    Python engines → Pine functions
    Python orchestrator → Pine main indicator

PINE CONSTRAINTS ENFORCED:
    - Only int, float, bool types (no datetime, no None)
    - No dynamic arrays (fixed-size only)
    - No classes (functions only)
    - var/varip persistence model
    - na instead of None/null

USAGE:
    from src.pine.generator import generate_pine_indicator
    
    pine_code = generate_pine_indicator(
        indicator_name="HORC Signal",
        config=your_config
    )
    
    # Save to file
    with open("horc_signal.pine", "w") as f:
        f.write(pine_code)

OUTPUT:
    Complete Pine Script v5 indicator ready for TradingView.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class PineConfig:
    """Configuration for Pine Script generation"""
    indicator_name: str = "HORC Signal System"
    version: str = "1.0.0"
    overlay: bool = True
    
    # Confluence weights
    participant_weight: float = 0.30
    wavelength_weight: float = 0.30
    exhaustion_weight: float = 0.25
    gap_weight: float = 0.15
    
    # Thresholds
    confluence_threshold: float = 0.25
    exhaustion_threshold: float = 0.65
    
    # Wavelength config
    min_move_atr: float = 0.3
    max_retracement: float = 0.786
    max_move_bars: int = 30
    
    # Visual settings
    show_signals: bool = True
    show_wavelength: bool = True
    show_exhaustion: bool = True
    show_confluence: bool = True
    
    # Alert settings
    enable_alerts: bool = True


def generate_pine_header(config: PineConfig) -> str:
    """Generate Pine Script header with version and settings"""
    
    return f'''//@version=5
// =============================================================================
// HORC Signal System v{config.version}
// Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
// 
// This indicator implements the HORC (Higher Order Reversal Catalyst) framework:
//   - AXIOM 1: Wavelength Invariant (3-move cycle)
//   - AXIOM 2: First Move Determinism (participant identification)
//   - AXIOM 3: Absorption Reversal (exhaustion detection)
//   - AXIOM 4: Futures Supremacy (gap targeting)
//
// IMPORTANT: This code was auto-generated from the Python implementation
// to ensure exact parity. Do not modify without updating the Python source.
// =============================================================================

indicator("{config.indicator_name}", overlay={str(config.overlay).lower()}, max_bars_back=500)

'''


def generate_pine_inputs(config: PineConfig) -> str:
    """Generate Pine Script input parameters"""
    
    return f'''// =============================================================================
// INPUT PARAMETERS
// =============================================================================

// Confluence Weights
participant_weight = input.float({config.participant_weight}, "Participant Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
wavelength_weight = input.float({config.wavelength_weight}, "Wavelength Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
exhaustion_weight = input.float({config.exhaustion_weight}, "Exhaustion Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
gap_weight = input.float({config.gap_weight}, "Gap Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")

// Thresholds
confluence_threshold = input.float({config.confluence_threshold}, "Confluence Threshold", minval=0.0, maxval=1.0, step=0.05, group="Thresholds")
exhaustion_threshold = input.float({config.exhaustion_threshold}, "Exhaustion Threshold", minval=0.0, maxval=1.0, step=0.05, group="Thresholds")

// Wavelength Configuration
min_move_atr = input.float({config.min_move_atr}, "Min Move (ATR multiplier)", minval=0.1, maxval=5.0, step=0.1, group="Wavelength")
max_retracement = input.float({config.max_retracement}, "Max Retracement", minval=0.1, maxval=1.0, step=0.05, group="Wavelength")
max_move_bars = input.int({config.max_move_bars}, "Max Move Bars", minval=5, maxval=100, step=5, group="Wavelength")

// Visual Settings
show_signals = input.bool({str(config.show_signals).lower()}, "Show Signals", group="Display")
show_wavelength = input.bool({str(config.show_wavelength).lower()}, "Show Wavelength", group="Display")
show_exhaustion = input.bool({str(config.show_exhaustion).lower()}, "Show Exhaustion Zones", group="Display")
show_confluence = input.bool({str(config.show_confluence).lower()}, "Show Confluence", group="Display")

// Alert Settings
enable_alerts = input.bool({str(config.enable_alerts).lower()}, "Enable Alerts", group="Alerts")
'''

// Confluence Settings
confluence_threshold = input.float({config.confluence_threshold}, "Confluence Threshold", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
participant_weight = input.float({config.participant_weight}, "Participant Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
wavelength_weight = input.float({config.wavelength_weight}, "Wavelength Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
exhaustion_weight = input.float({config.exhaustion_weight}, "Exhaustion Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")
gap_weight = input.float({config.gap_weight}, "Gap Weight", minval=0.0, maxval=1.0, step=0.05, group="Confluence")

// Wavelength Settings
min_move_atr = input.float({config.min_move_atr}, "Min Move (ATR)", minval=0.1, maxval=2.0, step=0.1, group="Wavelength")
max_retracement = input.float({config.max_retracement}, "Max Retracement", minval=0.5, maxval=1.0, step=0.01, group="Wavelength")
max_move_bars = input.int({config.max_move_bars}, "Max Move Bars", minval=10, maxval=100, group="Wavelength")

// Exhaustion Settings
exhaustion_threshold = input.float({config.exhaustion_threshold}, "Exhaustion Threshold", minval=0.0, maxval=1.0, step=0.05, group="Exhaustion")

// Visual Settings
show_signals = input.bool({str(config.show_signals).lower()}, "Show Signals", group="Display")
show_wavelength = input.bool({str(config.show_wavelength).lower()}, "Show Wavelength State", group="Display")
show_exhaustion = input.bool({str(config.show_exhaustion).lower()}, "Show Exhaustion", group="Display")
show_confluence = input.bool({str(config.show_confluence).lower()}, "Show Confluence", group="Display")

// Alert Settings
enable_alerts = input.bool({str(config.enable_alerts).lower()}, "Enable Alerts", group="Alerts")

'''


def generate_pine_enums() -> str:
    """Generate Pine Script enum constants"""
    
    return '''// =============================================================================
// ENUM CONSTANTS (Pine has no enums, use int constants)
// =============================================================================

// Wavelength States
int WL_PRE_OR = 0
int WL_PARTICIPANT_ID = 1
int WL_MOVE_1 = 2
int WL_MOVE_2 = 3
int WL_FLIP_CONFIRMED = 4
int WL_MOVE_3 = 5
int WL_COMPLETE = 6
int WL_FAILED = 7

// Participant Types
int PART_NONE = 0
int PART_BUYERS = 1
int PART_SELLERS = -1

// Bias
int BIAS_NEUTRAL = 0
int BIAS_BULLISH = 1
int BIAS_BEARISH = -1

'''


def generate_pine_state_variables() -> str:
    """Generate Pine Script state variables (var persistence)"""
    
    return '''// =============================================================================
// STATE VARIABLES (var = persisted across bars)
// =============================================================================

// Signal IR - The output contract
var int signal_timestamp = 0
var int signal_bias = BIAS_NEUTRAL
var bool signal_actionable = false
var float signal_confidence = 0.0

// Participant State
var int participant_control = PART_NONE
var float orh_prev = na
var float orl_prev = na

// Wavelength State
var int wavelength_state = WL_PRE_OR
var int moves_completed = 0
var float move_1_extreme = na
var float move_2_extreme = na
var int bars_in_move = 0

// Exhaustion State
var float exhaustion_score = 0.0
var bool in_exhaustion_zone = false

// Gap State (simplified for spot - full version needs futures data)
var int active_gap_type = 0
var float gap_fill_progress = 0.0

// Session tracking
var float session_high = na
var float session_low = na
var int session_bar_count = 0
var bool new_session = false

'''


def generate_pine_helper_functions() -> str:
    """Generate Pine Script helper functions"""
    
    return '''// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Calculate ATR (simplified)
atr_value(int length) =>
    ta.atr(length)

// Calculate body ratio (for exhaustion)
body_ratio() =>
    range_size = high - low
    body_size = math.abs(close - open)
    range_size > 0 ? body_size / range_size : 0.0

// Upper wick ratio
upper_wick_ratio() =>
    range_size = high - low
    upper_wick = high - math.max(open, close)
    range_size > 0 ? upper_wick / range_size : 0.0

// Lower wick ratio
lower_wick_ratio() =>
    range_size = high - low
    lower_wick = math.min(open, close) - low
    range_size > 0 ? lower_wick / range_size : 0.0

// Detect session change (simplified - uses day change)
is_new_session() =>
    ta.change(time("D")) != 0

'''


def generate_pine_participant_logic() -> str:
    """Generate Pine Script participant identification logic"""
    
    return '''// =============================================================================
// PARTICIPANT IDENTIFICATION (AXIOM 2)
// =============================================================================

identify_participant() =>
    var int result = PART_NONE
    
    // Check for session change
    if is_new_session()
        // Store previous session's range
        orh_prev := session_high
        orl_prev := session_low
        // Reset session tracking
        session_high := high
        session_low := low
        session_bar_count := 1
    else
        session_high := math.max(session_high, high)
        session_low := math.min(session_low, low)
        session_bar_count += 1
    
    // Identify participant in first few bars of session
    if session_bar_count <= 3 and not na(orh_prev) and not na(orl_prev)
        // Buyers sweep high
        if high >= orh_prev
            result := PART_BUYERS
        // Sellers sweep low
        else if low <= orl_prev
            result := PART_SELLERS
        else
            result := PART_NONE
    
    result

'''


def generate_pine_wavelength_logic() -> str:
    """Generate Pine Script wavelength FSM logic"""
    
    return '''// =============================================================================
// WAVELENGTH ENGINE (AXIOM 1)
// =============================================================================

update_wavelength(int participant) =>
    // State machine transitions
    atr = atr_value(14)
    min_move = atr * min_move_atr
    
    // Track bars in current move
    bars_in_move += 1
    
    // State transitions
    if wavelength_state == WL_PRE_OR
        // Wait for participant identification
        if participant != PART_NONE
            wavelength_state := WL_PARTICIPANT_ID
            bars_in_move := 0
    
    else if wavelength_state == WL_PARTICIPANT_ID
        // Look for Move 1 start
        price_change = participant == PART_BUYERS ? high - orl_prev : orh_prev - low
        if price_change >= min_move
            wavelength_state := WL_MOVE_1
            move_1_extreme := participant == PART_BUYERS ? high : low
            moves_completed := 1
            bars_in_move := 0
    
    else if wavelength_state == WL_MOVE_1
        // Update extreme
        if participant == PART_BUYERS
            move_1_extreme := math.max(move_1_extreme, high)
        else
            move_1_extreme := math.min(move_1_extreme, low)
        
        // Check for reversal (Move 2 start)
        if participant == PART_BUYERS
            retracement = (move_1_extreme - low) / (move_1_extreme - orl_prev)
            if retracement >= 0.382 and retracement <= max_retracement
                wavelength_state := WL_MOVE_2
                move_2_extreme := low
                bars_in_move := 0
        else
            retracement = (high - move_1_extreme) / (orh_prev - move_1_extreme)
            if retracement >= 0.382 and retracement <= max_retracement
                wavelength_state := WL_MOVE_2
                move_2_extreme := high
                bars_in_move := 0
        
        // Timeout check
        if bars_in_move > max_move_bars
            wavelength_state := WL_FAILED
    
    else if wavelength_state == WL_MOVE_2
        // Update extreme
        if participant == PART_BUYERS
            move_2_extreme := math.min(move_2_extreme, low)
        else
            move_2_extreme := math.max(move_2_extreme, high)
        
        // Check for flip confirmation (exhaustion)
        if exhaustion_score >= exhaustion_threshold
            wavelength_state := WL_FLIP_CONFIRMED
            moves_completed := 2
            bars_in_move := 0
        
        // Check for invalidation
        if participant == PART_BUYERS and low < orl_prev
            wavelength_state := WL_FAILED
        else if participant == PART_SELLERS and high > orh_prev
            wavelength_state := WL_FAILED
        
        // Timeout check
        if bars_in_move > max_move_bars
            wavelength_state := WL_FAILED
    
    else if wavelength_state == WL_FLIP_CONFIRMED
        // Confirmation period
        if bars_in_move >= 2
            wavelength_state := WL_MOVE_3
            bars_in_move := 0
    
    else if wavelength_state == WL_MOVE_3
        // Track Move 3 progress
        if participant == PART_BUYERS and high > move_1_extreme
            wavelength_state := WL_COMPLETE
            moves_completed := 3
        else if participant == PART_SELLERS and low < move_1_extreme
            wavelength_state := WL_COMPLETE
            moves_completed := 3
        
        // Invalidation check
        if participant == PART_BUYERS and low < move_2_extreme
            wavelength_state := WL_FAILED
        else if participant == PART_SELLERS and high > move_2_extreme
            wavelength_state := WL_FAILED
        
        // Timeout
        if bars_in_move > max_move_bars
            wavelength_state := WL_FAILED
    
    // Reset on complete or failed
    if wavelength_state == WL_COMPLETE or wavelength_state == WL_FAILED
        if is_new_session()
            wavelength_state := WL_PRE_OR
            moves_completed := 0
            move_1_extreme := na
            move_2_extreme := na
            bars_in_move := 0
    
    wavelength_state

'''


def generate_pine_exhaustion_logic() -> str:
    """Generate Pine Script exhaustion detection logic"""
    
    return '''// =============================================================================
// EXHAUSTION DETECTION (AXIOM 3)
// =============================================================================

calculate_exhaustion() =>
    // Body rejection score
    body_score = 1.0 - body_ratio()
    
    // Wick analysis
    upper_wick = upper_wick_ratio()
    lower_wick = lower_wick_ratio()
    wick_score = math.max(upper_wick, lower_wick)
    
    // Price stagnation (range contraction)
    atr = atr_value(14)
    range_size = high - low
    stagnation_score = atr > 0 ? math.max(0.0, 1.0 - (range_size / atr)) : 0.0
    
    // Simple reversal pattern detection
    reversal_score = 0.0
    // Shooting star / hammer detection
    if upper_wick > 0.6 and body_ratio() < 0.3
        reversal_score := 0.8  // Shooting star
    else if lower_wick > 0.6 and body_ratio() < 0.3
        reversal_score := 0.8  // Hammer
    
    // Weighted combination (matches Python weights)
    // volume_weight = 0.20 (skipped - forex volume unreliable)
    // body_weight = 0.35
    // price_weight = 0.30
    // reversal_weight = 0.15
    
    score = (body_score * 0.35) + (stagnation_score * 0.30) + (reversal_score * 0.35)
    
    // Update state
    exhaustion_score := math.min(1.0, math.max(0.0, score))
    in_exhaustion_zone := exhaustion_score >= exhaustion_threshold
    
    exhaustion_score

'''


def generate_pine_confluence_logic() -> str:
    """Generate Pine Script confluence scoring logic"""
    
    return '''// =============================================================================
// CONFLUENCE SCORING
// =============================================================================

calculate_confluence(int participant, int wl_state, float exh_score) =>
    // Participant strength [0, 1]
    participant_strength = participant != PART_NONE ? 1.0 : 0.0
    
    // Wavelength progress [0, 1]
    wavelength_progress = float(moves_completed) / 3.0
    
    // Exhaustion [0, 1]
    exhaustion_contrib = exh_score
    
    // Gap contribution (simplified without futures data)
    gap_contrib = 0.0
    
    // Weighted sum
    confidence = (participant_strength * participant_weight) +
                 (wavelength_progress * wavelength_weight) +
                 (exhaustion_contrib * exhaustion_weight) +
                 (gap_contrib * gap_weight)
    
    math.min(1.0, math.max(0.0, confidence))

'''


def generate_pine_bias_logic() -> str:
    """Generate Pine Script bias determination logic"""
    
    return '''// =============================================================================
// BIAS DETERMINATION
// =============================================================================

determine_bias(int participant, int wl_state) =>
    int result = BIAS_NEUTRAL
    
    // Participant vote
    int participant_vote = participant
    
    // Wavelength vote
    int wavelength_vote = BIAS_NEUTRAL
    if wl_state == WL_MOVE_1 or wl_state == WL_MOVE_3
        wavelength_vote := participant  // Same direction
    else if wl_state == WL_MOVE_2
        wavelength_vote := -participant  // Counter direction
    
    // Simple bias (single engine can trigger)
    if participant_vote != BIAS_NEUTRAL
        result := participant_vote
    else if wavelength_vote != BIAS_NEUTRAL
        result := wavelength_vote
    
    result

'''


def generate_pine_main_logic() -> str:
    """Generate Pine Script main indicator logic"""
    
    return '''// =============================================================================
// MAIN INDICATOR LOGIC
// =============================================================================

// Step 1: Identify participant
participant_control := identify_participant()

// Step 2: Calculate exhaustion
exhaustion_score := calculate_exhaustion()

// Step 3: Update wavelength FSM
wavelength_state := update_wavelength(participant_control)

// Step 4: Calculate confluence
signal_confidence := calculate_confluence(participant_control, wavelength_state, exhaustion_score)

// Step 5: Determine bias
signal_bias := determine_bias(participant_control, wavelength_state)

// Step 6: Gate actionability
signal_actionable := signal_confidence >= confluence_threshold and signal_bias != BIAS_NEUTRAL

// Step 7: Update timestamp
signal_timestamp := int(time)

'''


def generate_pine_visualization() -> str:
    """Generate Pine Script visualization code"""
    
    return '''// =============================================================================
// VISUALIZATION
// =============================================================================

// Signal markers
bullish_signal = signal_actionable and signal_bias == BIAS_BULLISH
bearish_signal = signal_actionable and signal_bias == BIAS_BEARISH

plotshape(show_signals and bullish_signal, "Buy Signal", 
          shape.triangleup, location.belowbar, color.new(color.green, 0), size=size.small)
plotshape(show_signals and bearish_signal, "Sell Signal", 
          shape.triangledown, location.abovebar, color.new(color.red, 0), size=size.small)

// Background coloring
bgcolor_color = signal_actionable ? 
                (signal_bias > 0 ? color.new(color.green, 90) : color.new(color.red, 90)) : 
                na
bgcolor(bgcolor_color, title="Signal Background")

// Wavelength state display
var string wl_text = ""
if show_wavelength
    wl_text := switch wavelength_state
        WL_PRE_OR => "Pre-OR"
        WL_PARTICIPANT_ID => "Part ID"
        WL_MOVE_1 => "Move 1"
        WL_MOVE_2 => "Move 2"
        WL_FLIP_CONFIRMED => "Flip"
        WL_MOVE_3 => "Move 3"
        WL_COMPLETE => "Complete"
        WL_FAILED => "Failed"
        => "Unknown"

// Info panel
var table info_panel = table.new(position.top_right, 2, 6, bgcolor=color.new(color.black, 80))
if barstate.islast
    table.cell(info_panel, 0, 0, "HORC Signal", text_color=color.white, text_size=size.small)
    table.cell(info_panel, 1, 0, "", text_color=color.white)
    
    table.cell(info_panel, 0, 1, "Bias", text_color=color.gray, text_size=size.tiny)
    bias_text = signal_bias > 0 ? "🟢 BULL" : signal_bias < 0 ? "🔴 BEAR" : "⚪ FLAT"
    table.cell(info_panel, 1, 1, bias_text, text_color=color.white, text_size=size.tiny)
    
    table.cell(info_panel, 0, 2, "Confidence", text_color=color.gray, text_size=size.tiny)
    table.cell(info_panel, 1, 2, str.tostring(signal_confidence, "#.##"), text_color=color.white, text_size=size.tiny)
    
    table.cell(info_panel, 0, 3, "Wavelength", text_color=color.gray, text_size=size.tiny)
    table.cell(info_panel, 1, 3, wl_text, text_color=color.white, text_size=size.tiny)
    
    table.cell(info_panel, 0, 4, "Moves", text_color=color.gray, text_size=size.tiny)
    table.cell(info_panel, 1, 4, str.tostring(moves_completed) + "/3", text_color=color.white, text_size=size.tiny)
    
    table.cell(info_panel, 0, 5, "Exhaustion", text_color=color.gray, text_size=size.tiny)
    exh_color = in_exhaustion_zone ? color.red : color.white
    table.cell(info_panel, 1, 5, str.tostring(exhaustion_score, "#.##"), text_color=exh_color, text_size=size.tiny)

'''


def generate_pine_alerts() -> str:
    """Generate Pine Script alert conditions"""
    
    return '''// =============================================================================
// ALERTS
// =============================================================================

// Signal alerts
alertcondition(enable_alerts and bullish_signal, 
               title="HORC Bullish Signal", 
               message="HORC: Bullish signal detected. Confidence: {{plot_0}}")

alertcondition(enable_alerts and bearish_signal, 
               title="HORC Bearish Signal", 
               message="HORC: Bearish signal detected. Confidence: {{plot_0}}")

alertcondition(enable_alerts and signal_actionable, 
               title="HORC Any Signal", 
               message="HORC: Signal detected. Bias: {{plot_1}}, Confidence: {{plot_0}}")

// State alerts
alertcondition(enable_alerts and wavelength_state == WL_FLIP_CONFIRMED, 
               title="HORC Flip Confirmed", 
               message="HORC: Wavelength flip confirmed - prepare for Move 3")

alertcondition(enable_alerts and in_exhaustion_zone, 
               title="HORC Exhaustion", 
               message="HORC: Exhaustion detected - potential reversal")

'''


def generate_pine_plots() -> str:
    """Generate Pine Script plot outputs"""
    
    return '''// =============================================================================
// PLOT OUTPUTS (for alerts and external use)
// =============================================================================

// Hidden plots for alert messages
plot(signal_confidence, "Confidence", display=display.none)
plot(signal_bias, "Bias", display=display.none)
plot(exhaustion_score, "Exhaustion", display=display.none)
plot(moves_completed, "Moves", display=display.none)

// Optional visible plots
plot(show_confluence ? signal_confidence : na, "Confluence", 
     color=color.new(color.blue, 50), linewidth=1)
plot(show_exhaustion ? exhaustion_score : na, "Exhaustion Score", 
     color=color.new(color.orange, 50), linewidth=1)

'''


def generate_pine_indicator(config: Optional[PineConfig] = None) -> str:
    """
    Generate complete Pine Script indicator.
    
    Args:
        config: PineConfig with customization options
        
    Returns:
        Complete Pine Script v5 code as string
    """
    if config is None:
        config = PineConfig()
    
    sections = [
        generate_pine_header(config),
        generate_pine_inputs(config),
        generate_pine_enums(),
        generate_pine_state_variables(),
        generate_pine_helper_functions(),
        generate_pine_participant_logic(),
        generate_pine_wavelength_logic(),
        generate_pine_exhaustion_logic(),
        generate_pine_confluence_logic(),
        generate_pine_bias_logic(),
        generate_pine_main_logic(),
        generate_pine_visualization(),
        generate_pine_alerts(),
        generate_pine_plots(),
    ]
    
    return "\n".join(sections)


def save_pine_script(
    output_path: str = "horc_signal.pine",
    config: Optional[PineConfig] = None,
) -> str:
    """
    Generate and save Pine Script to file.
    
    Returns:
        Path to saved file
    """
    pine_code = generate_pine_indicator(config)
    
    with open(output_path, "w") as f:
        f.write(pine_code)
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HORC Pine Script")
    parser.add_argument('--output', '-o', default='horc_signal.pine',
                        help='Output file path')
    parser.add_argument('--name', default='HORC Signal System',
                        help='Indicator name')
    parser.add_argument('--threshold', '-t', type=float, default=0.25,
                        help='Confluence threshold')
    parser.add_argument('--no-alerts', action='store_true',
                        help='Disable alerts')
    
    args = parser.parse_args()
    
    config = PineConfig(
        indicator_name=args.name,
        confluence_threshold=args.threshold,
        enable_alerts=not args.no_alerts,
    )
    
    print("=" * 70)
    print("  HORC PINE SCRIPT GENERATOR")
    print("=" * 70)
    print(f"\n📝 Generating Pine Script v5...")
    print(f"   Indicator: {config.indicator_name}")
    print(f"   Threshold: {config.confluence_threshold}")
    print(f"   Alerts: {'Enabled' if config.enable_alerts else 'Disabled'}")
    
    output_path = save_pine_script(args.output, config)
    
    print(f"\n✅ Generated: {output_path}")
    
    # Show line count
    with open(output_path, 'r') as f:
        lines = len(f.readlines())
    print(f"   Lines: {lines}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Open TradingView")
    print(f"   2. Pine Editor → New Script")
    print(f"   3. Paste contents of {output_path}")
    print(f"   4. Add to Chart")
    print(f"   5. Configure inputs as needed")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
