"""Authoritative planner factory for AdaptiveRL.

Provides unified creation and parameter validation for classical deterministic
and sampling-based motion planners.
"""

from __future__ import annotations

from typing import Any, Union

from adaptive_rl.config import AStarParametersConfig, RRTStarParametersConfig
from adaptive_rl.planners.astar import AStarPlanner
from adaptive_rl.planners.base import BasePlanner
from adaptive_rl.planners.rrt_star import RRTStarPlanner

PlannerType = Union[BasePlanner, AStarPlanner, RRTStarPlanner]


def make_planner(name: str, **params: Any) -> PlannerType:
    """Instantiate and validate a classical planner from its canonical identifier and parameters.

    Args:
        name: Planner algorithm name ('astar', 'rrt_star', or alias 'rrt*').
        **params: Planner-specific configuration parameters.

    Returns:
        Configured and validated planner instance.

    Raises:
        ValueError: If planner name is unknown, if plain 'rrt' is requested, or if
            parameters violate schema and relational validation constraints.
    """
    clean_name = str(name).strip().lower()

    if clean_name == "rrt":
        raise ValueError(
            "Planner name 'rrt' is not supported. Did you mean 'rrt_star'? "
            "Standard RRT does not perform tree rewiring and is not implemented."
        )

    if clean_name in ("rrt*", "rrt_star"):
        validated = RRTStarParametersConfig.model_validate(params).model_dump(exclude_none=False)
        return RRTStarPlanner(**validated)

    if clean_name == "astar":
        validated = AStarParametersConfig.model_validate(params).model_dump(exclude_none=False)
        return AStarPlanner(**validated)

    raise ValueError(
        f"Unknown planner '{name}'. Supported registered planners: 'astar', 'rrt_star'."
    )
