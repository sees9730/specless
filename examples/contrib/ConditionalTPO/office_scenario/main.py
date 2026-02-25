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


def main():
    print(f"\n=== Creating Office Environment === ")
    env, transition_system = create_office_environment(render_mode="rgb_array")

    print(f"\n=== Defining Events and Mapping === ")
    mapping = define_events_and_mapping(transition_system)

    print(f"\n=== Defining Regions === ")
    regions = define_regions(mapping)

    print(f"\n=== Creating Conditional TPOs === ")
    ctpo = create_conditional_tpos(mapping, regions)

    print(f"\n=== Building Unified TPO and TSP === ")
    unified_tpo, edges, tsp_nodes = build_tpo_and_tsp(mapping, ctpo)

    print(f"\n=== Scenario 1: Avoid Puddle (direct path cheaper) === ")
    solver_s1, tours_s1, cost_s1, _ = solve_with_costs(
        mapping, regions, unified_tpo, edges, tsp_nodes, ctpo, favor_puddle=False
    )

    print(f"\n=== Scenario 2: Through Puddle (puddle path cheaper) === ")
    solver_s2, tours_s2, cost_s2, _ = solve_with_costs(
        mapping, regions, unified_tpo, edges, tsp_nodes, ctpo, favor_puddle=True
    )

    print(f"\n=== Generating Videos === ")
    env_render, _ = create_office_environment(render_mode="rgb_array")
    n = mapping.obs_to_node
    event_labels = {
        n["initial_state0"]: "start",
        n["floor_blue"]:     "puddle",
        n["floor_grey"]:     "carpet",
        n["floor_yellow"]:   "charger",
    }

    generate_video_from_tour(
        env_render, tours_s1[0], mapping, transition_system,
        filename="visualization/scenario1_avoid_puddle.mp4",
    )
    generate_path_image(
        env_render, tours_s1[0], mapping, transition_system,
        filename="visualization/scenario1_avoid_puddle_path.pdf",
        event_labels=event_labels,
    )
    generate_video_from_tour(
        env_render, tours_s2[0], mapping, transition_system,
        filename="visualization/scenario2_through_puddle.mp4",
    )
    generate_path_image(
        env_render, tours_s2[0], mapping, transition_system,
        filename="visualization/scenario2_through_puddle_path.pdf",
        event_labels=event_labels,
    )
    env_render.close()


if __name__ == "__main__":
    main()
