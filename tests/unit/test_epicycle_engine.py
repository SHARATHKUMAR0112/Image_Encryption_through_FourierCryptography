"""
Unit tests for EpicycleEngine.

Tests the epicycle animation engine that computes rotating circle positions
for visualizing Fourier series reconstruction.
"""

import math

import numpy as np
import pytest

from fourier_encryption.core.epicycle_engine import EpicycleEngine
from fourier_encryption.models.data_models import FourierCoefficient


class TestEpicycleEngine:
    """Test suite for EpicycleEngine class."""
    
    def test_init_with_valid_coefficients(self):
        """Test initialization with valid coefficients."""
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        assert engine.coefficients == coefficients
    
    def test_init_with_empty_coefficients_raises_error(self):
        """Test that initialization with empty coefficients raises ValueError."""
        with pytest.raises(ValueError, match="coefficients list cannot be empty"):
            EpicycleEngine([])
    
    def test_compute_state_at_t_zero(self):
        """Test computing epicycle state at t=0."""
        # Create a simple coefficient: frequency=1, amplitude=2, phase=0
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=2.0,
                phase=0.0,
                complex_value=complex(2.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        state = engine.compute_state(0.0)
        
        assert state.time == 0.0
        assert len(state.positions) == 1
        assert state.positions[0] == complex(0, 0)  # First epicycle at origin
        # At t=0, angle = 1*0 + 0 = 0, so vector = 2*e^(i*0) = 2+0i
        assert abs(state.trace_point - complex(2.0, 0.0)) < 1e-10
    
    def test_compute_state_at_t_pi_over_2(self):
        """Test computing epicycle state at t=π/2."""
        # frequency=1, amplitude=2, phase=0
        # At t=π/2: angle = 1*(π/2) + 0 = π/2
        # vector = 2*e^(i*π/2) = 2*(cos(π/2) + i*sin(π/2)) = 0 + 2i
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=2.0,
                phase=0.0,
                complex_value=complex(2.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        state = engine.compute_state(math.pi / 2)
        
        assert abs(state.time - math.pi / 2) < 1e-10
        assert abs(state.trace_point - complex(0.0, 2.0)) < 1e-10
    
    def test_compute_state_with_multiple_coefficients(self):
        """Test computing state with multiple epicycles."""
        # Two coefficients: both at t=0 should add up
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            ),
            FourierCoefficient(
                frequency=1,
                amplitude=0.5,
                phase=0.0,
                complex_value=complex(0.5, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        state = engine.compute_state(0.0)
        
        assert len(state.positions) == 2
        assert state.positions[0] == complex(0, 0)  # First at origin
        assert abs(state.positions[1] - complex(1.0, 0.0)) < 1e-10  # Second after first
        # Total: 1.0 + 0.5 = 1.5
        assert abs(state.trace_point - complex(1.5, 0.0)) < 1e-10
    
    def test_compute_state_with_phase_offset(self):
        """Test computing state with phase offset."""
        # frequency=1, amplitude=1, phase=π/2
        # At t=0: angle = 1*0 + π/2 = π/2
        # vector = 1*e^(i*π/2) = 0 + 1i
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=math.pi / 2,
                complex_value=complex(0.0, 1.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        state = engine.compute_state(0.0)
        
        assert abs(state.trace_point - complex(0.0, 1.0)) < 1e-10
    
    def test_compute_state_invalid_time_negative(self):
        """Test that negative time raises ValueError."""
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        with pytest.raises(ValueError, match="time must be in"):
            engine.compute_state(-0.1)
    
    def test_compute_state_invalid_time_too_large(self):
        """Test that time > 2π raises ValueError."""
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        with pytest.raises(ValueError, match="time must be in"):
            engine.compute_state(2 * math.pi + 0.1)
    
    def test_generate_animation_frames_count(self):
        """Test that generate_animation_frames produces correct number of frames."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        frames = list(engine.generate_animation_frames(num_frames=10, speed=1.0))
        
        assert len(frames) == 10
    
    def test_generate_animation_frames_time_progression(self):
        """Test that frames progress from 0 to 2π."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        frames = list(engine.generate_animation_frames(num_frames=4, speed=1.0))
        
        # Should have times approximately: 0, π/2, π, 3π/2
        assert abs(frames[0].time - 0.0) < 1e-10
        assert abs(frames[1].time - math.pi / 2) < 1e-10
        assert abs(frames[2].time - math.pi) < 1e-10
        assert abs(frames[3].time - 3 * math.pi / 2) < 1e-10
    
    def test_generate_animation_frames_with_speed_multiplier(self):
        """Test animation with speed multiplier."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Speed 2.0 should complete two full rotations
        frames = list(engine.generate_animation_frames(num_frames=8, speed=2.0))
        
        assert len(frames) == 8
        # All frames should still have valid time values in [0, 2π]
        for frame in frames:
            assert 0 <= frame.time <= 2 * math.pi
    
    def test_generate_animation_frames_slow_speed(self):
        """Test animation with slow speed (< 1.0)."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Speed 0.5 should complete half a rotation over num_frames
        frames = list(engine.generate_animation_frames(num_frames=4, speed=0.5))
        
        assert len(frames) == 4
        # With speed 0.5, we go from 0 to π (half of 2π)
        # Last frame should be close to π but not exactly (endpoint=False)
        # The frames are: 0, π/4, π/2, 3π/4
        assert frames[-1].time < math.pi  # Should be less than π
        assert frames[-1].time > math.pi / 2  # Should be more than π/2
    
    def test_generate_animation_frames_invalid_num_frames(self):
        """Test that invalid num_frames raises ValueError."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        with pytest.raises(ValueError, match="num_frames must be positive"):
            list(engine.generate_animation_frames(num_frames=0, speed=1.0))
        
        with pytest.raises(ValueError, match="num_frames must be positive"):
            list(engine.generate_animation_frames(num_frames=-5, speed=1.0))
    
    def test_generate_animation_frames_invalid_speed_too_low(self):
        """Test that speed < 0.1 raises ValueError."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        with pytest.raises(ValueError, match="speed must be in"):
            list(engine.generate_animation_frames(num_frames=10, speed=0.05))
    
    def test_generate_animation_frames_invalid_speed_too_high(self):
        """Test that speed > 10.0 raises ValueError."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        with pytest.raises(ValueError, match="speed must be in"):
            list(engine.generate_animation_frames(num_frames=10, speed=15.0))
    
    def test_epicycle_radius_equals_amplitude(self):
        """Test that epicycle radius equals coefficient amplitude (Property 6)."""
        # Create coefficients with known amplitudes
        coefficients = [
            FourierCoefficient(
                frequency=0,
                amplitude=3.0,
                phase=0.0,
                complex_value=complex(3.0, 0.0)
            ),
            FourierCoefficient(
                frequency=1,
                amplitude=2.0,
                phase=0.0,
                complex_value=complex(2.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        state = engine.compute_state(0.0)
        
        # Verify that the distance from each position to the next equals the amplitude
        # First epicycle: from origin to positions[1]
        radius_1 = abs(state.positions[1] - state.positions[0])
        assert abs(radius_1 - coefficients[0].amplitude) < 1e-10
        
        # Second epicycle: from positions[1] to trace_point
        radius_2 = abs(state.trace_point - state.positions[1])
        assert abs(radius_2 - coefficients[1].amplitude) < 1e-10
    
    def test_pause_sets_paused_state(self):
        """Test that pause() sets the engine to paused state."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Initially not paused
        assert not engine.is_paused()
        
        # Pause the engine
        engine.pause()
        
        # Should now be paused
        assert engine.is_paused()
    
    def test_resume_clears_paused_state(self):
        """Test that resume() clears the paused state."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Pause then resume
        engine.pause()
        assert engine.is_paused()
        
        engine.resume()
        assert not engine.is_paused()
    
    def test_reset_clears_paused_state_and_time(self):
        """Test that reset() clears paused state and resets time to 0."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Advance time and pause
        engine.compute_state(math.pi)
        engine.pause()
        
        assert engine.is_paused()
        assert abs(engine.get_current_time() - math.pi) < 1e-10
        
        # Reset
        engine.reset()
        
        # Should be unpaused and at time 0
        assert not engine.is_paused()
        assert abs(engine.get_current_time() - 0.0) < 1e-10
    
    def test_pause_preserves_current_time(self):
        """Test that pausing preserves the current animation time."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Advance to a specific time
        t = math.pi / 4
        engine.compute_state(t)
        
        # Pause
        engine.pause()
        
        # Time should be preserved
        assert abs(engine.get_current_time() - t) < 1e-10
    
    def test_resume_from_paused_state(self):
        """Test that animation can resume from paused state."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Advance, pause, then resume
        t1 = math.pi / 3
        engine.compute_state(t1)
        engine.pause()
        engine.resume()
        
        # Should be able to continue from where we left off
        t2 = math.pi / 2
        state = engine.compute_state(t2)
        
        assert not engine.is_paused()
        assert abs(state.time - t2) < 1e-10
    
    def test_multiple_pause_resume_cycles(self):
        """Test multiple pause/resume cycles."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # First cycle
        engine.pause()
        assert engine.is_paused()
        engine.resume()
        assert not engine.is_paused()
        
        # Second cycle
        engine.pause()
        assert engine.is_paused()
        engine.resume()
        assert not engine.is_paused()
    
    def test_reset_from_middle_of_animation(self):
        """Test resetting from the middle of an animation."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Advance to middle of animation
        engine.compute_state(math.pi)
        assert abs(engine.get_current_time() - math.pi) < 1e-10
        
        # Reset
        engine.reset()
        
        # Should be back at start
        assert abs(engine.get_current_time() - 0.0) < 1e-10
        assert not engine.is_paused()
    
    def test_generate_frames_completes_full_rotation(self):
        """Test that generate_animation_frames completes a full rotation."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Generate frames for full rotation
        num_frames = 100
        frames = list(engine.generate_animation_frames(num_frames=num_frames, speed=1.0))
        
        # Should have correct number of frames
        assert len(frames) == num_frames
        
        # First frame should be at t=0
        assert abs(frames[0].time - 0.0) < 1e-10
        
        # Last frame should be close to 2π (but not exactly, due to endpoint=False)
        # With endpoint=False, last frame is at 2π * (num_frames-1) / num_frames
        expected_last_time = 2 * math.pi * (num_frames - 1) / num_frames
        assert abs(frames[-1].time - expected_last_time) < 1e-10
        
        # All frames should have valid time values
        for frame in frames:
            assert 0 <= frame.time <= 2 * math.pi
    
    def test_generate_frames_with_different_speeds(self):
        """Test frame generation with different speed settings."""
        coefficients = [
            FourierCoefficient(
                frequency=1,
                amplitude=1.0,
                phase=0.0,
                complex_value=complex(1.0, 0.0)
            )
        ]
        engine = EpicycleEngine(coefficients)
        
        # Test slow speed (0.5x)
        frames_slow = list(engine.generate_animation_frames(num_frames=10, speed=0.5))
        assert len(frames_slow) == 10
        # With speed 0.5, we cover π radians (half rotation)
        # Last frame at π * 9/10 = 0.9π
        assert frames_slow[-1].time < math.pi
        
        # Test normal speed (1.0x)
        frames_normal = list(engine.generate_animation_frames(num_frames=10, speed=1.0))
        assert len(frames_normal) == 10
        # With speed 1.0, we cover 2π radians (full rotation)
        # Last frame at 2π * 9/10 = 1.8π
        assert frames_normal[-1].time > math.pi
        
        # Test fast speed (2.0x)
        frames_fast = list(engine.generate_animation_frames(num_frames=10, speed=2.0))
        assert len(frames_fast) == 10
        # With speed 2.0, we cover 4π radians (two full rotations)
        # But times are wrapped to [0, 2π], so all should be valid
        for frame in frames_fast:
            assert 0 <= frame.time <= 2 * math.pi

