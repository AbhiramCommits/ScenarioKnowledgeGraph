"""Synthetic driving log generator."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from skg.logs.schema import AgentTrack, LaneSegment, ScenarioMeta, TrafficControl


# Behavior taxonomy with descriptions
BEHAVIORS = {
    "LEAD_VEHICLE_BRAKING": "Lead vehicle sudden deceleration from {speed} m/s to {final_speed} m/s over {duration} s",
    "CUT_IN": "Vehicle cuts in from {side} lane with {ttc} s TTC",
    "UNPROTECTED_LEFT_TURN": "Ego makes unprotected left turn across {oncoming_count} oncoming vehicles",
    "PEDESTRIAN_JAYWALK": "Pedestrian jaywalks in front of ego at {speed} m/s, {dist} m away",
    "MERGE_CONFLICT": "Two vehicles merging into same lane with {gap} m gap at {relative_speed} m/s",
    "OCCLUDED_CROSSING": "Pedestrian appears from behind parked car {dist} m ahead",
    "STOP_SIGN_NEGOTIATION": "Ego approaches stop sign with {other_count} agents in intersection",
    "FREE_FLOW": "Free-flowing traffic on highway with {n_agents} agents, average speed {avg_speed} m/s",
}


def generate_scenario(
    scenario_id: str,
    behavior: str,
    n_agents: int,
    duration_steps: int,
    fps: int = 10,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, list[LaneSegment], list[TrafficControl]]:
    """Generate single scenario."""
    if rng is None:
        rng = np.random.default_rng()

    dt = 1.0 / fps
    times = np.arange(duration_steps) * dt
    tracks = []
    min_ttc = float("inf")
    max_speed = 0.0

    # Generate ego
    ego_x = np.linspace(0, 50 + rng.uniform(-5, 5), duration_steps)
    ego_y = np.linspace(0, rng.normal(0, 2), duration_steps)
    ego_vx = np.diff(ego_x, prepend=ego_x[0]) / dt
    ego_vy = np.diff(ego_y, prepend=ego_y[0]) / dt
    ego_heading = np.arctan2(ego_vy, ego_vx)
    ego_speed = np.sqrt(ego_vx**2 + ego_vy**2)
    max_speed = max(max_speed, np.max(ego_speed))

    for t_idx, t_ms in enumerate((times * 1000).astype(int)):
        tracks.append(
            AgentTrack(
                scenario_id=scenario_id,
                timestamp_ms=int(t_ms),
                agent_id="ego",
                agent_type="EGO",
                x=float(ego_x[t_idx]),
                y=float(ego_y[t_idx]),
                heading=float(ego_heading[t_idx]),
                velocity_x=float(ego_vx[t_idx]),
                velocity_y=float(ego_vy[t_idx]),
                lane_id="lane_0",
                is_ego=True,
            )
        )

    # Generate other agents based on behavior
    for agent_idx in range(1, n_agents):
        agent_id = f"agent_{agent_idx}"
        agent_type = rng.choice(["VEHICLE", "PEDESTRIAN", "CYCLIST"], p=[0.7, 0.2, 0.1])

        if behavior == "LEAD_VEHICLE_BRAKING" and agent_idx == 1:
            # Lead vehicle that brakes suddenly
            brake_step = int(0.4 * duration_steps)
            speed_initial = rng.uniform(10, 15)
            speed_final = rng.uniform(2, 5)
            speeds = np.linspace(speed_initial, speed_final, duration_steps)
            y_offset = rng.uniform(1.5, 3.0)
            x_base = np.linspace(20, 60, duration_steps)
        elif behavior == "CUT_IN" and agent_idx == 1:
            # Vehicle cutting in from side
            x_base = np.linspace(15, 40, duration_steps)
            lane_change_start = int(0.3 * duration_steps)
            lane_change_end = int(0.7 * duration_steps)
            y_offset = np.linspace(3.0, -1.8, duration_steps)
            speeds = np.full(duration_steps, rng.uniform(12, 16))
            # Calculate TTC at closest point
            min_gap = min(np.abs(x_base - ego_x))
            if min_gap > 0:
                min_ttc = min(min_ttc, min_gap / max(5.0, np.mean(speeds)))
        else:
            # Random traffic
            speeds = np.full(duration_steps, rng.uniform(5, 18))
            x_base = np.linspace(rng.uniform(-20, 0), rng.uniform(30, 60), duration_steps)
            y_offset = rng.uniform(-4, 4)

        max_speed = max(max_speed, np.max(speeds))
        vx = np.diff(x_base, prepend=x_base[0]) / dt
        vy = np.zeros_like(vx)
        if isinstance(y_offset, np.ndarray):
            vy = np.diff(y_offset, prepend=y_offset[0]) / dt
            y = y_offset
        else:
            y = np.full(duration_steps, y_offset)

        heading = np.arctan2(vy, vx)

        for t_idx, t_ms in enumerate((times * 1000).astype(int)):
            tracks.append(
                AgentTrack(
                    scenario_id=scenario_id,
                    timestamp_ms=int(t_ms),
                    agent_id=agent_id,
                    agent_type=agent_type,
                    x=float(x_base[t_idx]),
                    y=float(y[t_idx]) if isinstance(y, np.ndarray) else float(y),
                    heading=float(heading[t_idx]),
                    velocity_x=float(vx[t_idx]),
                    velocity_y=float(vy[t_idx]),
                    lane_id=f"lane_{agent_idx % 3}",
                    is_ego=False,
                )
            )

    # Create lane segments
    lanes = [
        LaneSegment(
            scenario_id=scenario_id,
            lane_id=f"lane_{i}",
            centerline_x=list(np.linspace(0, 100, 20)),
            centerline_y=list(np.full(20, float(i * 3.7))),
            successor_id=f"lane_{i+1}" if i < 2 else None,
            predecessor_id=f"lane_{i-1}" if i > 0 else None,
            speed_limit=25.0,
        )
        for i in range(3)
    ]

    # Create traffic controls
    controls = [
        TrafficControl(
            scenario_id=scenario_id,
            control_id=f"control_{i}",
            control_type=rng.choice(["STOP", "SIGNAL", "YIELD"]),
            lane_id=f"lane_{i}",
            x=float(rng.uniform(30, 70)),
            y=float(i * 3.7),
        )
        for i in range(2)
    ]

    # Description
    if behavior == "LEAD_VEHICLE_BRAKING":
        desc = BEHAVIORS[behavior].format(speed=10, final_speed=3, duration=3)
    elif behavior == "CUT_IN":
        desc = BEHAVIORS[behavior].format(side="left", ttc=2.5)
    elif behavior == "PEDESTRIAN_JAYWALK":
        desc = BEHAVIORS[behavior].format(speed=1.5, dist=15)
    else:
        desc = BEHAVIORS[behavior].format(
            n_agents=n_agents, avg_speed=max_speed * 0.7, oncoming_count=1, gap=5, relative_speed=3, dist=20, other_count=1
        )

    if min_ttc == float("inf"):
        min_ttc = 10.0

    return (
        pd.DataFrame([t.__dict__ for t in tracks]),
        lanes,
        controls,
    )


def generate_corpus(
    n_scenarios: int,
    out_dir: str | Path,
    seed: int = 7,
    n_agents: int = 7,
    duration_steps: int = 50,
    fps: int = 10,
) -> None:
    """Generate synthetic corpus."""
    out_dir = Path(out_dir)
    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    all_tracks = []
    scenario_summary = []
    total_rows = 0

    behaviors = list(BEHAVIORS.keys())

    for scenario_idx in range(n_scenarios):
        scenario_id = f"scenario_{scenario_idx:06d}"
        behavior = behaviors[scenario_idx % len(behaviors)]
        n_agents_this = rng.integers(max(2, min(4, n_agents - 1)), n_agents + 1)

        tracks_df, lanes, controls = generate_scenario(
            scenario_id=scenario_id,
            behavior=behavior,
            n_agents=n_agents_this,
            duration_steps=duration_steps,
            fps=fps,
            rng=rng,
        )

        all_tracks.append(tracks_df)
        total_rows += len(tracks_df)

        meta = ScenarioMeta(
            scenario_id=scenario_id,
            behavior_label=behavior,
            description=f"{behavior} scenario with {n_agents_this} agents",
            n_agents=n_agents_this,
            duration_steps=duration_steps,
            min_ttc=rng.uniform(2.0, 8.0),
            max_speed=rng.uniform(15, 25),
        )
        scenario_summary.append(meta.__dict__)

        if (scenario_idx + 1) % max(1, n_scenarios // 10) == 0:
            print(f"  Generated {scenario_idx + 1}/{n_scenarios} scenarios")

    # Write tracks parquet
    tracks_df_all = pd.concat(all_tracks, ignore_index=True)
    table = pa.Table.from_pandas(tracks_df_all)
    pq.write_table(table, scenarios_dir / "part-000.parquet")

    # Write summary
    summary_df = pd.DataFrame(scenario_summary)
    summary_table = pa.Table.from_pandas(summary_df)
    pq.write_table(summary_table, out_dir / "scenario_summary.parquet")

    print(f"Generated {n_scenarios} scenarios with {total_rows} track points")
    print(f"Output: {scenarios_dir}")
    print(f"Summary: {out_dir / 'scenario_summary.parquet'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="Number of scenarios")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.n} scenarios...")
    generate_corpus(args.n, args.out, args.seed)
