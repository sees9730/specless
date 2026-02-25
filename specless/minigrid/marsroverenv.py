import gymnasium as gym
from gym_minigrid.minigrid import (
    Floor,
    Goal,
    Grid,
    Lava,
    MiniGridEnv,
    MissionSpace,
)


class MarsRoverEnvBase(MiniGridEnv):
    """Base class for Mars Rover environments.

    Grid dimensions: 16 cells (x) by 7 cells (y) [includes walls]
    Interior dimensions: 14 cells (x) by 5 cells (y)

    Base mission regions (shared by both variants):
    ─────────────────────────────────────────────────────────────────────────
    Observation      Object        Event  Description
    ─────────────────────────────────────────────────────────────────────────
    floor_green      Floor(green)  e1     Science zone entry
    floor_red        Floor(red)    e2     Site A — soil heating pad
    floor_purple     Floor(purple) e3     Site A — outgassing / atmospheric sensor
    floor_blue       Floor(blue)   e4     Site B — atmospheric reading
    floor_yellow     Floor(yellow) e5     Lander / depot (return point)
    lava_red         Lava()        --     Geological outcrop region (orange background)
    floor_grey       Floor(grey)   e6     Arrive at outcrop  [conditional, inside lava region]
    goal_green       Goal()        e7     Collect rock sample  [conditional, inside lava region]
    ─────────────────────────────────────────────────────────────────────────

    Base layout (interior, x=1..14, y=1..6, origin top-left):

        col:  1  2  3  4  5  6  7  8  9 10 11 12 13 14
        row1: .  .  .  .  .  .  .  R  P  .  .  .  .  .  
        row2: .  .  .  .  .  .  .  .  .  .  .  .  .  .  
        row3: .  .  .  .  G  .  .  .  .  .  .  Y  .  .  
        row4: .  .  .  .  .  .  .  .  .  .  .  .  .  .  
        row5: .  .  .  S  .  .  .  B  .  .  .  .  .  . 

    Agent start: (4, 5)
    """

    def __init__(self, width=16, height=7, agent_start_pos=(4, 5),
                 agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        super().__init__(
            mission_space=MissionSpace(lambda: "Complete Mars atmospheric science mission"),
            width=width,
            height=height,
            max_steps=4 * width * height,
            see_through_walls=True,
            agent_view_size=99,
            **kwargs,
        )

    def _place_base_objects(self):
        # e1 — Science Zone entry (floor_green)  x=5, y=3
        self.put_obj(Floor("green"), 5, 3)

        # e2 — Site A heating pad (floor_red)  x=8, y=1
        self.put_obj(Floor("red"), 8, 1)

        # e3 — Site A outgassing sensor (floor_purple)  x=9, y=1
        self.put_obj(Floor("purple"), 9, 1)

        # e4 — Site B atmospheric sensor (floor_blue)  x=8, y=5
        self.put_obj(Floor("blue"), 8, 5)

        # e5 — Lander (floor_yellow)  x=12, y=3
        self.put_obj(Floor("yellow"), 12, 3)

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self._place_base_objects()
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = "Complete Mars atmospheric science mission"


class MarsRoverEnvDirect(MarsRoverEnvBase):
    """Mars Rover — outcrop region top-left; never on the optimal path.

    A 3x3 lava patch at x=1..3, y=1..3 forms the outcrop region, with
    e6 and e7 inside it.  The mission path goes right and never touches it.

        col:  1  2  3  4  5  6  7  8  9 10 11 12 13 14
        row1: Lv Lv Lv .  .  .  .  R  P  .  .  .  .  .  
        row2: Lv Og Gg .  .  .  .  .  .  .  .  .  .  . 
        row3: Lv Lv Lv .  G  .  .  .  .  .  .  Y  .  . 
        row4: .  .  .  .  .  .  .  .  .  .  .  .  .  . 
        row5: .  .  .  S  .  .  .  B  .  .  .  .  .  .

    """

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self._place_base_objects()

        # Lava region — 3x3 patch top-left: x=1..3, y=1..3
        for x in range(1, 4):
            for y in range(1, 4):
                self.put_obj(Lava(), x, y)
        # e6 — arrive at outcrop (floor_grey) inside lava region
        self.put_obj(Floor("grey"), 2, 2)
        # e7 — collect sample (goal_green) inside lava region
        self.put_obj(Goal(), 3, 2)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = "Complete Mars atmospheric science mission"


class MarsRoverEnvOutcrop(MarsRoverEnvBase):
    """Mars Rover — outcrop in the lower corridor between Site B and the Lander.

    A lava corridor at x=10..11, y=5 sits directly between e4 (x=8, y=5)
    and e5 (x=12, y=3).  Any path from e4 to e5 via the lower row must cross
    the lava, activating z=1 automatically.  e6 and e7 are event tiles inside.

        col:  1  2  3  4  5  6  7  8  9 10 11 12 13 14
        row1: .  .  .  .  .  .  .  R  P  .  .  .  .  .
        row2: .  .  .  .  .  .  .  .  .  .  Lv .  .  .
        row3: .  .  .  .  G  .  .  .  .  .  Lv  Y Lv .
        row4: .  .  .  .  .  .  .  .  .  .  Lv Og Gg .
        row5: .  .  .  S  .  .  .  B  .  . .   .  .  .
    """

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self._place_base_objects()

        # Lava surrounds e5 at (12,3): close all entries except through e6/e7
        for y in range(2, 5):
            self.put_obj(Lava(), 11, y)   # left column: (11,2), (11,3), (11,4)
        self.put_obj(Lava(), 12, 2)       # top: (12,2)
        self.put_obj(Lava(), 13, 2)       # top-right: (13,2)
        self.put_obj(Lava(), 13, 3)       # right: (13,3)
        self.put_obj(Lava(), 12, 4)       # bottom: (12,4)

        # e6 — arrive at outcrop (floor_grey) at (12,2)
        self.put_obj(Floor("grey"), 12, 4)
        # e7 — collect sample (goal_green) at (13,2)
        self.put_obj(Goal(), 13, 4)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        self.mission = "Complete Mars atmospheric science mission"


# Register both environments
gym.register(
    id="MiniGrid-MarsRover-Direct-v0",
    entry_point="specless.minigrid.marsroverenv:MarsRoverEnvDirect",
)
gym.register(
    id="MiniGrid-MarsRover-Outcrop-v0",
    entry_point="specless.minigrid.marsroverenv:MarsRoverEnvOutcrop",
)
