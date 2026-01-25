from office_scenario import (
    create_office_environment,
    define_events_and_mapping,
    define_regions,
    create_conditional_tpos
)

def main():
    # First create the office environment and transition system
    print(f"\n=== Creating Office Environment === ")
    env, transition_system = create_office_environment(render_mode="rgb_array")

    # Define mandatory and optional events and create the regions
    print(f"\n=== Defining Events and Mapping === ")
    mapping = define_events_and_mapping(transition_system)

    print(f"\n=== Defining Regions === ")
    regions = define_regions(mapping)

    # Create conditional TPOs (1 base + 2 cTPOs)
    print(f"\n=== Creating Conditional TPOs === ")
    ctpo = create_conditional_tpos(mapping, regions)



if __name__ == "__main__":
    main()