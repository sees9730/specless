"""
Mars Rover Atmospheric Science Mission — main entry point.

Same conditional TPO structure, two DTS environments.  Costs are real grid
shortest-path distances — no manual tuning needed.

The conditional TPO has two branches (mirroring the office puddle scenario):
  - IF outcrop visited  → enforce e6 → e7 [0,30] and e7 → e5 [0,30]
  - IF outcrop NOT visited → enforce e5 global deadline [0,50]

  Scenario 1 — MiniGrid-MarsRover-Direct-v0:
    Outcrop (e6, e7) is a column at x=1 (far left).  It is never on the
    shortest path from e4 to e5 → solver sets z=0, skips the outcrop.

  Scenario 2 — MiniGrid-MarsRover-Outcrop-v0:
    Outcrop (e6, e7) is a full-height column at x=10-11, directly between
    e4 (x=8-9) and e5 (x=12-13).  Every path from e4 to e5 must cross it →
    solver sets z=1, conditional TPO activates automatically.

Run from this directory:
    python main.py

    # With RViz2 replay (scenario 2 by default):
    python main.py --rviz
    python main.py --rviz --rviz-scenario 1
"""

import argparse
import warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*already in registry.*")

from mars_rover_scenario import (
    create_mars_environment,
    define_events_and_mapping,
    define_regions,
    create_conditional_tpos,
    build_tpo_and_tsp,
    solve,
)
from specless.utils.tour_video import generate_video_from_tour, generate_path_image


def run_scenario(env_id, label, video_filename, rviz=False, publisher_script=None,
                 json_path="visualization/tour_data.json"):
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    env, transition_system = create_mars_environment(env_id, render_mode="rgb_array")
    mapping    = define_events_and_mapping(transition_system)
    regions    = define_regions(mapping)
    ctpo       = create_conditional_tpos(mapping, regions)
    unified_tpo, base_tpo, edges, tsp_nodes = build_tpo_and_tsp(mapping, ctpo)

    solver, tours, cost, _ = solve(
        mapping, transition_system, unified_tpo, base_tpo, edges, tsp_nodes, ctpo,
    )

    n = mapping.obs_to_node
    event_labels = {
        n["initial_state0"]: "depot",
        n["floor_green"]:    "e1 (science zone)",
        n["floor_red"]:      "e2 (soil heating)",
        n["floor_purple"]:   "e3 (outgassing)",
        n["floor_blue"]:     "e4 (atmospheric)",
        n["floor_yellow"]:   "e5 (lander)",
    }
    if "floor_grey" in n:
        event_labels[n["floor_grey"]] = "e6 (outcrop)"
    if "goal_green" in n:
        event_labels[n["goal_green"]] = "e7 (sample)"

    env_render, _ = create_mars_environment(env_id, render_mode="rgb_array")
    generate_video_from_tour(
        env_render, tours[0], mapping, transition_system,
        filename=f"visualization/{video_filename}",
    )
    pdf_filename = video_filename.replace(".mp4", "_path.pdf")
    generate_path_image(
        env_render, tours[0], mapping, transition_system,
        filename=f"visualization/{pdf_filename}",
        event_labels=event_labels,
    )
    env_render.close()

    if rviz:
        from rviz_visualizer import visualize_tour_rviz
        visualize_tour_rviz(
            tours[0], mapping, transition_system,
            event_labels=event_labels,
            json_path=json_path,
            publisher_script=publisher_script,
        )


def main():
    parser = argparse.ArgumentParser(description="Mars Rover Mission Planner")
    parser.add_argument("--rviz", action="store_true",
                        help="After solving, launch RViz2 tour replay (blocking, Ctrl-C to stop)")
    parser.add_argument("--rviz-scenario", choices=["1", "2"], default="2",
                        help="Which scenario to replay in RViz2 (default: 2)")
    args = parser.parse_args()

    run_scenario(
        env_id="MiniGrid-MarsRover-Direct-v0",
        label="Scenario 1: Outcrop far left — detour too costly, skipped",
        video_filename="scenario1_direct_mission.mp4",
        rviz=args.rviz and args.rviz_scenario == "1",
        publisher_script="rviz_publisher.py",
        json_path="visualization/tour_data_s1.json",
    )
    run_scenario(
        
        env_id="MiniGrid-MarsRover-Outcrop-v0",
        label="Scenario 2: Outcrop below lander — cheap detour, conditional TPO activates",
        video_filename="scenario2_outcrop_mission.mp4",
        rviz=args.rviz and args.rviz_scenario == "2",
        publisher_script="rviz_publisher_s2.py",
        json_path="visualization/tour_data_s2.json",
    )


if __name__ == "__main__":
    main()
