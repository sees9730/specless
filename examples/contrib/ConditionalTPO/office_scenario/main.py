"""
Office Environment — main entry point.

Two scenarios with different travel costs select whether the robot goes
through the puddle or avoids it:

  Scenario 1 — avoid puddle (direct path to charger is cheaper)
  Scenario 2 — through puddle (puddle → carpet → charger path is cheaper)

Run from this directory:
    python main.py

    # With RViz2 replay (scenario 1 by default):
    python main.py --rviz
    python main.py --rviz --rviz-scenario 2
"""

import argparse
import warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*already in registry.*")

from office_scenario import (
    create_office_environment,
    define_events_and_mapping,
    define_regions,
    create_conditional_tpos,
    build_tpo_and_tsp,
    solve_with_costs,
)
from specless.utils.tour_video import generate_video_from_tour, generate_path_image


def run_scenario(label, video_filename, favor_puddle, rviz=False,
                 publisher_script=None, json_path="visualization/tour_data.json"):
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    env, transition_system = create_office_environment(render_mode="rgb_array")
    mapping    = define_events_and_mapping(transition_system)
    regions    = define_regions(mapping)
    ctpo       = create_conditional_tpos(mapping, regions)
    unified_tpo, base_tpo, edges, tsp_nodes = build_tpo_and_tsp(mapping, ctpo)

    solver, tours, cost, _ = solve_with_costs(
        mapping, regions, unified_tpo, base_tpo, edges, tsp_nodes, ctpo,
        favor_puddle=favor_puddle,
    )

    n = mapping.obs_to_node
    event_labels = {
        n["initial_state0"]: "start",
        n["floor_blue"]:     "puddle",
        n["floor_grey"]:     "carpet",
        n["floor_yellow"]:   "charger",
    }

    env_render, _ = create_office_environment(render_mode="rgb_array")
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
    parser = argparse.ArgumentParser(description="Office Robot Mission Planner")
    parser.add_argument("--rviz", action="store_true",
                        help="After solving, launch RViz2 tour replay (blocking, Ctrl-C to stop)")
    parser.add_argument("--rviz-scenario", choices=["1", "2"], default="2",
                        help="Which scenario to replay in RViz2 (default: 1)")
    args = parser.parse_args()

    run_scenario(
        label="Scenario 1: Avoid Puddle (direct path cheaper)",
        video_filename="scenario1_avoid_puddle.mp4",
        favor_puddle=False,
        rviz=args.rviz and args.rviz_scenario == "1",
        publisher_script="rviz_publisher.py",
        json_path="visualization/tour_data_s1.json",
    )
    run_scenario(
        label="Scenario 2: Through Puddle (puddle path cheaper)",
        video_filename="scenario2_through_puddle.mp4",
        favor_puddle=True,
        rviz=args.rviz and args.rviz_scenario == "2",
        publisher_script="rviz_publisher.py",
        json_path="visualization/tour_data_s2.json",
    )


if __name__ == "__main__":
    main()
