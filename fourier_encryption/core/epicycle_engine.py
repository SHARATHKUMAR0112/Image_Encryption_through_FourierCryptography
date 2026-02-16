"""
Epicycle animation engine for visualizing Fourier series reconstruction.

This module implements the epicycle-based animation system that visualizes
how rotating circles (epicycles) combine to draw the original contour.
"""

import math
from typing import Iterator, List

import numpy as np

from fourier_encryption.models.data_models import EpicycleState, FourierCoefficient


class EpicycleEngine:
    """
    Compute epicycle positions for animation.

    The epicycle engine takes a list of Fourier coefficients and computes
    the positions of rotating circles (epicycles) at any time t. Each epicycle
    rotates at a frequency proportional to its coefficient's frequency, with
    a radius equal to the coefficient's amplitude.

    The animation traces out the original contour as the epicycles rotate
    through a full period (t: 0 → 2π).

    Attributes:
        coefficients: List of Fourier coefficients defining the epicycles
        _is_paused: Whether the animation is currently paused
        _current_time: Current time in the animation (for pause/resume)
    """

    def __init__(self, coefficients: List[FourierCoefficient]):
        """
        Initialize the epicycle engine with Fourier coefficients.

        Args:
            coefficients: List of FourierCoefficient objects

        Raises:
            ValueError: If coefficients list is empty
        """
        if not coefficients:
            raise ValueError("coefficients list cannot be empty")

        self.coefficients = coefficients
        self._is_paused = False
        self._current_time = 0.0

    def pause(self) -> None:
        """
        Pause the animation at the current time.

        When paused, the animation state is preserved and can be resumed
        from the same point.
        """
        self._is_paused = True

    def resume(self) -> None:
        """
        Resume the animation from the paused state.

        If the animation was not paused, this has no effect.
        """
        self._is_paused = False

    def reset(self) -> None:
        """
        Reset the animation to the beginning (t=0).

        This also unpauses the animation if it was paused.
        """
        self._current_time = 0.0
        self._is_paused = False

    def is_paused(self) -> bool:
        """
        Check if the animation is currently paused.

        Returns:
            True if paused, False otherwise
        """
        return self._is_paused

    def get_current_time(self) -> float:
        """
        Get the current time in the animation.

        Returns:
            Current time value in [0, 2π]
        """
        return self._current_time

    def compute_state(self, t: float) -> EpicycleState:
        """
        Compute epicycle positions at time t.

        Each epicycle is a rotating vector in the complex plane:
        - Position = previous_position + radius * e^(i*(frequency*t + phase))
        - The first epicycle starts at the origin
        - Each subsequent epicycle starts at the tip of the previous one

        Args:
            t: Time parameter in [0, 2π]

        Returns:
            EpicycleState containing time, positions, and trace point

        Raises:
            ValueError: If t is not in [0, 2π]
        """
        if not (0 <= t <= 2 * math.pi):
            raise ValueError(f"time must be in [0, 2π], got {t}")

        # Update current time
        self._current_time = t

        positions: List[complex] = []
        current_position = complex(0, 0)

        # Compute each epicycle position
        for coeff in self.coefficients:
            # Store the center position of this epicycle
            positions.append(current_position)

            # Compute the rotation angle: frequency * t + phase
            angle = coeff.frequency * t + coeff.phase

            # Compute the vector from this epicycle: radius * e^(i*angle)
            vector = coeff.amplitude * complex(math.cos(angle), math.sin(angle))

            # Move to the tip of this epicycle
            current_position += vector

        # The final position is the trace point being drawn
        trace_point = current_position

        return EpicycleState(
            time=t,
            positions=positions,
            trace_point=trace_point
        )

    def generate_animation_frames(
        self,
        num_frames: int,
        speed: float = 1.0
    ) -> Iterator[EpicycleState]:
        """
        Generate frames for full rotation (t: 0 → 2π).

        Creates an iterator that yields epicycle states for a complete animation
        cycle. The speed parameter controls how fast the animation progresses.

        Args:
            num_frames: Number of frames to generate for one complete rotation
            speed: Animation speed multiplier in [0.1, 10.0]
                  - speed < 1.0: slower animation
                  - speed = 1.0: normal speed (one rotation per cycle)
                  - speed > 1.0: faster animation

        Yields:
            EpicycleState for each frame in the animation

        Raises:
            ValueError: If num_frames <= 0 or speed not in [0.1, 10.0]
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        if not (0.1 <= speed <= 10.0):
            raise ValueError(f"speed must be in [0.1, 10.0], got {speed}")

        # Generate time values from 0 to 2π
        # Speed affects how much of the rotation we complete
        t_values = np.linspace(0, 2 * math.pi * speed, num_frames, endpoint=False)

        for t in t_values:
            # Wrap t to [0, 2π] range to handle speed > 1.0
            t_wrapped = t % (2 * math.pi)
            yield self.compute_state(t_wrapped)


