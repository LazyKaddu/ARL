"""4-Way Signalized Traffic Intersection Gymnasium environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from adaptive_rl.environments.base import AdaptiveRLEnv
from adaptive_rl.environments.traffic.simulation import (
    PHASE_NAMES,
    Approach,
    Phase,
    TrafficIntersection,
)


class TrafficSignalEnv(AdaptiveRLEnv[np.ndarray, int]):
    """Gymnasium-compatible 4-way signalized intersection environment.

    Controls green signal allocations across intersecting arterial and side roads to
    minimize vehicle queue lengths, waiting times, and delay while preventing excessive
    signal switching.

    Actions:
        Discrete(2):
            0: NORTH_SOUTH Green (East & West Red)
            1: EAST_WEST Green   (North & South Red)

    Observations:
        Box(10,) in [0.0, 1.0]:
            [0]: Normalized North queue  (queue_N / max_queue)
            [1]: Normalized South queue  (queue_S / max_queue)
            [2]: Normalized East queue   (queue_E / max_queue)
            [3]: Normalized West queue   (queue_W / max_queue)
            [4]: Normalized North wait   (max_wait_N / max_wait_limit)
            [5]: Normalized South wait   (max_wait_S / max_wait_limit)
            [6]: Normalized East wait    (max_wait_E / max_wait_limit)
            [7]: Normalized West wait    (max_wait_W / max_wait_limit)
            [8]: Current Signal Phase    (0.0 = NORTH_SOUTH, 1.0 = EAST_WEST)
            [9]: Phase Duration Ratio    (min(duration / max_phase_duration, 1.0))

    Rewards:
        Combined penalty/incentive formulation:
            - (queue_penalty_weight * sum(queues))
            - (wait_penalty_weight * max(wait_times))
            - (switch_penalty if switched else 0.0)
            - (premature_switch_penalty if switched before min_green_steps else 0.0)
            + (departure_reward * step_departures)
    """

    metadata = {"render_modes": ["ansi", "human"]}

    ACTION_NAMES: Dict[int, str] = {
        0: "NS_GREEN",
        1: "EW_GREEN",
    }

    def __init__(
        self,
        arrival_rates: Tuple[float, float, float, float] = (0.3, 0.3, 0.2, 0.2),
        departure_rate: int = 2,
        max_queue: int = 30,
        max_wait_limit: int = 100,
        max_phase_duration: int = 30,
        min_green_steps: int = 2,
        max_steps: int = 100,
        queue_penalty_weight: float = 0.2,
        wait_penalty_weight: float = 0.05,
        switch_penalty: float = 1.0,
        premature_switch_penalty: float = 2.0,
        departure_reward: float = 0.5,
        overflow_penalty: float = 20.0,
        terminate_on_overflow: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the Traffic Signal environment.

        Args:
            arrival_rates: Poisson arrival lambda parameters for (North, South, East, West).
            departure_rate: Saturation departure capacity per green approach per step.
            max_queue: Queue capacity per approach before overflow.
            max_wait_limit: Normalization upper bound for vehicle waiting time.
            max_phase_duration: Normalization upper bound for green phase duration.
            min_green_steps: Minimum green duration before phase switch avoids premature penalty.
            max_steps: Maximum steps per episode before truncation.
            queue_penalty_weight: Penalty scaling for cumulative queued vehicles.
            wait_penalty_weight: Penalty scaling for maximum vehicle waiting time.
            switch_penalty: Penalty incurred when switching active signal phases.
            premature_switch_penalty: Additional penalty for switching before min_green_steps.
            departure_reward: Positive reinforcement per vehicle successfully cleared.
            overflow_penalty: Penalty applied when an approach queue overflows.
            terminate_on_overflow: Whether queue overflow terminates the episode immediately.
            render_mode: Rendering mode ('ansi' or 'human').
        """
        super().__init__()
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if min_green_steps < 1:
            raise ValueError(f"min_green_steps must be >= 1, got {min_green_steps}")

        self.arrival_rates = arrival_rates
        self.departure_rate = departure_rate
        self.max_queue = max_queue
        self.max_wait_limit = max_wait_limit
        self.max_phase_duration = max_phase_duration
        self.min_green_steps = min_green_steps
        self.max_steps = max_steps
        self.queue_penalty_weight = queue_penalty_weight
        self.wait_penalty_weight = wait_penalty_weight
        self.switch_penalty = switch_penalty
        self.premature_switch_penalty = premature_switch_penalty
        self.departure_reward = departure_reward
        self.overflow_penalty = overflow_penalty
        self.terminate_on_overflow = terminate_on_overflow
        self.render_mode = render_mode

        # Define Farama Gymnasium action and observation spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.intersection = TrafficIntersection(
            arrival_rates=arrival_rates,
            departure_rate=departure_rate,
            max_queue=max_queue,
            max_wait_limit=max_wait_limit,
            min_green_steps=min_green_steps,
        )
        self._current_step = 0

    def _get_obs(self) -> np.ndarray:
        """Construct normalized 10-dimensional observation vector."""
        approaches = self.intersection.approaches
        q_n = approaches[Approach.NORTH].queue_length / float(self.max_queue)
        q_s = approaches[Approach.SOUTH].queue_length / float(self.max_queue)
        q_e = approaches[Approach.EAST].queue_length / float(self.max_queue)
        q_w = approaches[Approach.WEST].queue_length / float(self.max_queue)

        w_n = approaches[Approach.NORTH].max_wait_time / float(self.max_wait_limit)
        w_s = approaches[Approach.SOUTH].max_wait_time / float(self.max_wait_limit)
        w_e = approaches[Approach.EAST].max_wait_time / float(self.max_wait_limit)
        w_w = approaches[Approach.WEST].max_wait_time / float(self.max_wait_limit)

        phase_val = 0.0 if self.intersection.current_phase == Phase.NORTH_SOUTH else 1.0
        dur_val = min(1.0, float(self.intersection.phase_duration) / float(self.max_phase_duration))

        raw_obs = [q_n, q_s, q_e, q_w, w_n, w_s, w_e, w_w, phase_val, dur_val]
        clipped = np.clip(raw_obs, 0.0, 1.0).astype(np.float32)
        return clipped

    def _get_info(self) -> Dict[str, Any]:
        """Construct comprehensive inspection and metrics dictionary."""
        approaches = self.intersection.approaches
        queues = [
            approaches[Approach.NORTH].queue_length,
            approaches[Approach.SOUTH].queue_length,
            approaches[Approach.EAST].queue_length,
            approaches[Approach.WEST].queue_length,
        ]
        max_waits = [
            approaches[Approach.NORTH].max_wait_time,
            approaches[Approach.SOUTH].max_wait_time,
            approaches[Approach.EAST].max_wait_time,
            approaches[Approach.WEST].max_wait_time,
        ]
        mean_waits = [
            approaches[Approach.NORTH].mean_wait_time,
            approaches[Approach.SOUTH].mean_wait_time,
            approaches[Approach.EAST].mean_wait_time,
            approaches[Approach.WEST].mean_wait_time,
        ]
        total_q = sum(queues)

        return {
            "step": self._current_step,
            "max_steps": self.max_steps,
            "current_phase": int(self.intersection.current_phase),
            "phase_name": PHASE_NAMES[self.intersection.current_phase],
            "phase_duration": self.intersection.phase_duration,
            "total_switches": self.intersection.total_switches,
            "queues": queues,
            "total_queue": total_q,
            "max_wait": max(max_waits),
            "mean_wait": float(np.mean(mean_waits)),
            "cumulative_departures": self.intersection.cumulative_departures,
            "cumulative_arrivals": self.intersection.cumulative_arrivals,
            "cumulative_delay": self.intersection.cumulative_delay,
            "success": total_q <= 8,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the traffic intersection environment to initial state."""
        super().reset(seed=seed, options=options)
        self._current_step = 0

        # Optional parameter overrides via reset options
        rates = self.arrival_rates
        init_queues = None
        if options:
            if "arrival_rates" in options:
                rates = options["arrival_rates"]
            if "initial_queues" in options:
                init_queues = options["initial_queues"]

        self.intersection.reset(arrival_rates=rates, initial_queues=init_queues)

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), self._get_info()

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation time step."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Valid actions are 0 (NS) and 1 (EW).")

        act_int = (
            int(np.asarray(action).item())
            if isinstance(action, (np.ndarray, np.generic))
            else int(action)
        )

        self._current_step += 1

        # Check if switch is premature before advancing state
        was_premature = (
            act_int != int(self.intersection.current_phase)
            and self.intersection.phase_duration < self.min_green_steps
        )

        # Advance physics / queuing dynamics
        telemetry = self.intersection.step(act_int, self.np_random)

        # Compute multi-objective reward
        total_q = telemetry.total_queue
        max_w = max(telemetry.max_wait_times.values())

        queue_cost = self.queue_penalty_weight * float(total_q)
        wait_cost = self.wait_penalty_weight * float(max_w)
        switch_cost = self.switch_penalty if telemetry.phase_switched else 0.0
        premature_cost = self.premature_switch_penalty if was_premature else 0.0
        throughput_bonus = self.departure_reward * float(telemetry.total_departures)

        reward = float(throughput_bonus - queue_cost - wait_cost - switch_cost - premature_cost)

        terminated = False
        if telemetry.overflow:
            reward -= self.overflow_penalty
            if self.terminate_on_overflow:
                terminated = True

        truncated = self._current_step >= self.max_steps and not terminated

        info = self._get_info()
        info["action_taken"] = act_int
        info["action_name"] = self.ACTION_NAMES[act_int]
        info["phase_switched"] = telemetry.phase_switched
        info["step_departures"] = telemetry.total_departures
        info["step_arrivals"] = telemetry.total_arrivals
        info["overflow"] = telemetry.overflow
        info["premature_switch"] = was_premature
        info["success"] = bool(truncated and not telemetry.overflow)

        if self.render_mode == "human":
            print(self.render())

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> Optional[str]:
        """Render textual representation of the 4-way intersection layout."""
        ns_active = self.intersection.current_phase == Phase.NORTH_SOUTH
        sig_n = "[G]" if ns_active else "[R]"
        sig_s = "[G]" if ns_active else "[R]"
        sig_e = "[R]" if ns_active else "[G]"
        sig_w = "[R]" if ns_active else "[G]"

        q_n = self.intersection.approaches[Approach.NORTH].queue_length
        q_s = self.intersection.approaches[Approach.SOUTH].queue_length
        q_e = self.intersection.approaches[Approach.EAST].queue_length
        q_w = self.intersection.approaches[Approach.WEST].queue_length

        w_n = self.intersection.approaches[Approach.NORTH].max_wait_time
        w_s = self.intersection.approaches[Approach.SOUTH].max_wait_time
        w_e = self.intersection.approaches[Approach.EAST].max_wait_time
        w_w = self.intersection.approaches[Approach.WEST].max_wait_time

        phase_str = "NORTH_SOUTH (Green)" if ns_active else "EAST_WEST (Green)"

        lines = [
            "+--------------------------------------------------+",
            "|   4-WAY SIGNALIZED INTERSECTION OPTIMIZATION     |",
            "+--------------------------------------------------+",
            f"| Phase: {phase_str:<20} Step: {self._current_step:03d}/{self.max_steps:03d} |",
            f"| Duration: {self.intersection.phase_duration:02d} | Switches: {self.intersection.total_switches:02d} | Cleared: {self.intersection.cumulative_departures:03d}      |",
            "+--------------------------------------------------+",
            "                 |   N   |                          ",
            f"                 | Q:{q_n:02d}  | (Wait: {w_n:02d})             ",
            f"                 |  {sig_n}  |                          ",
            "  ---------------+       +---------------           ",
            f"   W  Q:{q_w:02d}  {sig_w}              {sig_e}  Q:{q_e:02d}  E    ",
            f"  (Wait: {w_w:02d})                       (Wait: {w_e:02d})     ",
            "  ---------------+       +---------------           ",
            f"                 |  {sig_s}  |                          ",
            f"                 | Q:{q_s:02d}  | (Wait: {w_s:02d})             ",
            "                 |   S   |                          ",
            "+--------------------------------------------------+",
            f"  Queues: [N={q_n}, S={q_s}, E={q_e}, W={q_w}] | Total: {q_n + q_s + q_e + q_w:02d}",
            "+--------------------------------------------------+",
        ]
        render_str = "\n".join(lines)

        if self.render_mode == "human":
            print(render_str)
            return None
        return render_str
