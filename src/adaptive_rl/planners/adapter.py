"""Planner evaluation adapter for AdaptiveRL.

Provides a unified interface for evaluating deterministic and sampling-based planners
(A*, RRT*) using metrics compatible with the RL evaluation framework, allowing side-by-side
comparison of planners and RL agents across discrete (GridWorld) and continuous (Navigation2D) domains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Union, cast

import numpy as np

if TYPE_CHECKING:
    from adaptive_rl.environments.gridworld.grid import GridWorldEnv
    from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv

from adaptive_rl.planners.astar import AStarPlanner
from adaptive_rl.planners.base import (
    BasePlanner,
    ContinuousCoordinate,
    GridCoordinate,
    PlannerResult,
)
from adaptive_rl.planners.rrt_star import RRTStarPlanner


@dataclass
class PlannerEvaluationMetrics:
    """Standardized evaluation metrics for deterministic and sampling-based planners.

    These metrics are comparable to RL evaluation metrics where meaningful.
    Planner-specific metrics (path_length, planning_time) are not applicable
    to RL algorithms, so they are kept separate from :class:`EvaluationMetrics`.

    Attributes:
        episodes: Total evaluation episodes executed.
        success_rate: Fraction of episodes in which the goal was reached.
        mean_path_length: Mean path length (steps or Euclidean distance) over successes.
        std_path_length: Standard deviation of path length over successes.
        min_path_length: Minimum path length over successes.
        max_path_length: Maximum path length over successes.
        mean_planning_time: Mean wall-clock planning time in seconds.
        std_planning_time: Standard deviation of planning time.
        collision_rate: Should always be 0.0 for a correct planner (validated paths only).
        all_path_lengths: Raw per-episode path lengths (0 if failed).
        all_planning_times: Raw per-episode planning times.
    """

    episodes: int
    success_rate: float = 0.0
    mean_path_length: Optional[float] = None
    std_path_length: Optional[float] = None
    min_path_length: Optional[float] = None
    max_path_length: Optional[float] = None
    mean_planning_time: float = 0.0
    std_planning_time: float = 0.0
    collision_rate: Optional[float] = None  # None if no dynamic environment execution occurred
    all_path_lengths: List[float] = field(default_factory=list)
    all_planning_times: List[float] = field(default_factory=list)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to a plain dictionary (JSON-serializable)."""
        return {
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "mean_path_length": self.mean_path_length,
            "std_path_length": self.std_path_length,
            "min_path_length": self.min_path_length,
            "max_path_length": self.max_path_length,
            "mean_planning_time": self.mean_planning_time,
            "std_planning_time": self.std_planning_time,
            "collision_rate": self.collision_rate,
            "all_path_lengths": self.all_path_lengths,
            "all_planning_times": self.all_planning_times,
            **self.additional_metrics,
        }


class PlannerAdapter:
    """Unified evaluator for classical baselines (A*, RRT*) on compatible environments.

    Supports:
    - :class:`AStarPlanner` evaluated on :class:`GridWorldEnv`
    - :class:`RRTStarPlanner` evaluated on :class:`ContinuousNavigation2DEnv`

    Example::

        from adaptive_rl.environments.gridworld.grid import GridWorldEnv
        from adaptive_rl.planners.astar import AStarPlanner
        from adaptive_rl.planners.adapter import PlannerAdapter

        env = GridWorldEnv()
        planner = AStarPlanner()
        adapter = PlannerAdapter(planner=planner, env=env)
        metrics = adapter.evaluate(num_episodes=20, base_seed=42)
    """

    def __init__(
        self,
        planner: Union[BasePlanner, AStarPlanner, RRTStarPlanner],
        env: Union[GridWorldEnv, ContinuousNavigation2DEnv],
    ) -> None:
        """Initialize the planner adapter.

        Args:
            planner: Planner instance (AStarPlanner or RRTStarPlanner).
            env: Compatible Gymnasium environment instance.

        Raises:
            TypeError: If planner/env types are incompatible or unsupported.
        """
        try:
            from adaptive_rl.environments.gridworld.grid import GridWorldEnv
            from adaptive_rl.environments.navigation.navigation2d import ContinuousNavigation2DEnv
        except ImportError as e:
            raise ImportError(
                f"PlannerAdapter requires the 'rl' optional dependencies to interact with Gymnasium environments: {e}. "
                "Install with: pip install 'adaptive-rl[rl]'"
            ) from e

        if isinstance(planner, (BasePlanner, AStarPlanner)):
            if not isinstance(env, GridWorldEnv):
                raise TypeError(f"AStarPlanner requires a GridWorldEnv, got {type(env).__name__}.")
        elif isinstance(planner, RRTStarPlanner):
            if not isinstance(env, ContinuousNavigation2DEnv):
                raise TypeError(
                    f"RRTStarPlanner requires a ContinuousNavigation2DEnv, got {type(env).__name__}."
                )
        else:
            raise TypeError(
                f"Unsupported planner type '{type(planner).__name__}'. "
                "Expected AStarPlanner or RRTStarPlanner."
            )

        self.planner = planner
        self.env = env

    def evaluate(
        self,
        num_episodes: int = 10,
        base_seed: Optional[int] = None,
    ) -> PlannerEvaluationMetrics:
        """Evaluate the planner across multiple seeded episodes.

        Args:
            num_episodes: Number of episodes to evaluate.
            base_seed: Base random seed. Episode i uses seed base_seed + i.

        Returns:
            PlannerEvaluationMetrics: Aggregated performance metrics.

        Raises:
            ValueError: If num_episodes < 1.
        """
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}.")

        if isinstance(self.planner, (BasePlanner, AStarPlanner)):
            return self._evaluate_gridworld(num_episodes, base_seed)
        elif isinstance(self.planner, RRTStarPlanner):
            return self._evaluate_continuous_nav(num_episodes, base_seed)
        else:
            raise RuntimeError(f"Unexpected planner state: {type(self.planner)}")

    def _evaluate_gridworld(
        self,
        num_episodes: int,
        base_seed: Optional[int],
    ) -> PlannerEvaluationMetrics:
        """Evaluate A* on GridWorldEnv."""
        env: GridWorldEnv = self.env  # type: ignore[assignment]
        planner: Union[BasePlanner, AStarPlanner] = self.planner  # type: ignore[assignment]

        successes = 0
        path_lengths: List[float] = []
        planning_times: List[float] = []
        successful_path_lengths: List[float] = []

        for ep in range(num_episodes):
            seed = base_seed + ep if base_seed is not None else None
            _obs, info = env.reset(seed=seed)

            start: GridCoordinate = info.get("agent_pos", env._agent_pos)
            goal: GridCoordinate = info.get("goal_pos", env._goal_pos)
            obstacles: Set[GridCoordinate] = set(env._obstacles)

            try:
                result: PlannerResult = planner.plan(
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                    width=env.width,
                    height=env.height,
                )
            except ValueError as exc:
                result = PlannerResult(success=False, failure_reason=str(exc))

            planning_times.append(result.planning_time_seconds)

            valid = (
                result.success
                and len(result.path) >= 1
                and result.path[0] == start
                and result.path[-1] == goal
                and AStarPlanner.validate_path(
                    cast(List[GridCoordinate], result.path), obstacles, env.width, env.height
                )
            )

            if valid:
                successes += 1
                path_lengths.append(float(result.path_length))
                successful_path_lengths.append(float(result.path_length))
            else:
                path_lengths.append(0.0)

        return self._build_metrics(
            num_episodes=num_episodes,
            successes=successes,
            path_lengths=path_lengths,
            successful_path_lengths=successful_path_lengths,
            planning_times=planning_times,
            env_name="gridworld",
            base_seed=base_seed,
        )

    def _evaluate_continuous_nav(
        self,
        num_episodes: int,
        base_seed: Optional[int],
    ) -> PlannerEvaluationMetrics:
        """Evaluate RRT* on ContinuousNavigation2DEnv."""
        env: ContinuousNavigation2DEnv = self.env  # type: ignore[assignment]
        planner: RRTStarPlanner = self.planner  # type: ignore[assignment]

        successes = 0
        path_lengths: List[float] = []
        planning_times: List[float] = []
        successful_path_lengths: List[float] = []

        for ep in range(num_episodes):
            seed = base_seed + ep if base_seed is not None else None
            _obs, info = env.reset(seed=seed)

            start = (float(info["agent_pos"][0]), float(info["agent_pos"][1]))
            goal = (float(info["goal_pos"][0]), float(info["goal_pos"][1]))
            obstacles = list(info.get("obstacles", env._obstacles))

            try:
                result = planner.plan(
                    start=start,
                    goal=goal,
                    arena_width=env.arena_width,
                    arena_height=env.arena_height,
                    obstacles=obstacles,
                    agent_radius=env.agent_radius,
                    goal_radius=env.goal_radius,
                    seed_override=seed,
                )
            except ValueError as exc:
                result = PlannerResult(success=False, failure_reason=str(exc))

            planning_times.append(result.planning_time_seconds)

            goal_reached = False
            if result.success and len(result.path) >= 1:
                dist_to_goal = math.hypot(
                    result.path[-1][0] - goal[0], result.path[-1][1] - goal[1]
                )
                goal_reached = (result.path[-1] == goal) or (dist_to_goal <= env.goal_radius)

            valid = (
                result.success
                and len(result.path) >= 1
                and result.path[0] == start
                and goal_reached
                and planner.validate_path(
                    cast(List[ContinuousCoordinate], result.path),
                    env.arena_width,
                    env.arena_height,
                    obstacles,
                    env.agent_radius,
                )
            )

            if valid:
                successes += 1
                path_lengths.append(float(result.path_length))
                successful_path_lengths.append(float(result.path_length))
            else:
                path_lengths.append(0.0)

        return self._build_metrics(
            num_episodes=num_episodes,
            successes=successes,
            path_lengths=path_lengths,
            successful_path_lengths=successful_path_lengths,
            planning_times=planning_times,
            env_name="navigation",
            base_seed=base_seed,
        )

    def _build_metrics(
        self,
        num_episodes: int,
        successes: int,
        path_lengths: List[float],
        successful_path_lengths: List[float],
        planning_times: List[float],
        env_name: str,
        base_seed: Optional[int],
    ) -> PlannerEvaluationMetrics:
        """Construct PlannerEvaluationMetrics dataclass."""
        success_rate = successes / num_episodes
        mean_path = float(np.mean(successful_path_lengths)) if successful_path_lengths else None
        std_path = float(np.std(successful_path_lengths)) if successful_path_lengths else None
        min_path = float(np.min(successful_path_lengths)) if successful_path_lengths else None
        max_path = float(np.max(successful_path_lengths)) if successful_path_lengths else None
        mean_pt = float(np.mean(planning_times)) if planning_times else 0.0
        std_pt = float(np.std(planning_times)) if planning_times else 0.0

        return PlannerEvaluationMetrics(
            episodes=num_episodes,
            success_rate=success_rate,
            mean_path_length=mean_path,
            std_path_length=std_path,
            min_path_length=min_path,
            max_path_length=max_path,
            mean_planning_time=mean_pt,
            std_planning_time=std_pt,
            collision_rate=None,
            all_path_lengths=path_lengths,
            all_planning_times=planning_times,
            additional_metrics={
                "planner": self.planner.name,
                "environment": env_name,
                "base_seed": base_seed,
                "successful_episodes": successes,
                "collision_semantics": "collision_rate is None because no dynamic environment execution occurred.",
            },
        )
