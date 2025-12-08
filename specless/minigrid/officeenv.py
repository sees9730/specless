import gymnasium as gym
from gym_minigrid.minigrid import (
    COLORS,
    Floor,
    Grid,
    MiniGridEnv,
    MissionSpace,
    fill_coords,
    point_in_rect,
)

class OfficeEnv(MiniGridEnv):
    """Office Environment with regions for different floor types.

    Grid dimensions: 12 cells (x) by 8 cells (y) [includes walls]
    Interior dimensions: 10 cells (x) by 6 cells (y)
    Each cell: 0.5 x 0.5 units
    Physical dimensions: 5 units (x) by 3 units (y)

    Regions:
    - Puddle (blue)
    - Carpet (gray)
    - Charger (yellow)
    """

    def __init__(
        self,
        width=12,
        height=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        **kwargs,
    ):
        """Initialize the Office Environment.

        Args:
            width: Number of cells in x-direction (default: 12, includes walls)
            height: Number of cells in y-direction (default: 8, includes walls)
            agent_start_pos: Starting position of the agent in grid cells (default: (1, 1))
            agent_start_dir: Starting direction (0=right, 1=down, 2=left, 3=up)
            **kwargs: Additional arguments for MiniGridEnv
        """
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            mission_space=MissionSpace(lambda: "Navigate the office environment"),
            width=width,
            height=height,
            max_steps=4 * width * height,
            see_through_walls=True,
            agent_view_size=99,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        """Generate the grid with walls and floor regions."""
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Puddle region (blue) - wet floor
        for x in range(5, 7):
            for y in range(2, 7):
                self.put_obj(Floor("blue"), x, y)

        # Carpet region (grey) - dry floor
        for x in range(9, 11):
            for y in range(5, 7):
                self.put_obj(Floor("grey"), x, y)

        # Charger region (yellow) - charging station
        for x in range(9, 11):
            for y in range(3, 5):
                self.put_obj(Floor("yellow"), x, y)

        # Place the agent at starting position
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Navigate the office environment"


# Register the environment
gym.register(
    id="MiniGrid-Office-v0",
    entry_point="specless.minigrid.officeenv:OfficeEnv",
)
