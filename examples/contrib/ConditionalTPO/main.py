from office_scenario import (
    create_office_environment,
    define_events_and_mapping,
    define_regions
)

def main():
    # First create the office environment and transition system
    env, transition_system = create_office_environment(render_mode="rgb_array")

    # Define mandatory and optional events and create the regions
    mapping = define_events_and_mapping(transition_system)
    regions = define_regions(mapping)



if __name__ == "__main__":
    main()