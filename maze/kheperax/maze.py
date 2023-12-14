from __future__ import annotations

from typing import List

import flax.struct
import jax.tree_util
from jax import numpy as jnp

from kheperax.geoms import Segment, Pos


@flax.struct.dataclass
class Maze:
    walls: Segment

    @classmethod
    def create(cls, segments_list: List[Segment] = None):
        if segments_list is None:
            segments_list = []

        # add borders
        segments_list.append(Segment(Pos(0, 0), Pos(0, 1)))
        segments_list.append(Segment(Pos(0, 1), Pos(1, 1)))
        segments_list.append(Segment(Pos(1, 1), Pos(1, 0)))
        segments_list.append(Segment(Pos(1, 0), Pos(0, 0)))

        walls = jax.tree_util.tree_map(
            lambda *x: jnp.asarray(x, dtype=jnp.float32), *segments_list
        )

        return Maze(walls)

    @classmethod
    def create_default_maze(cls):
        return cls.create(segments_list=[
            Segment(Pos(0.25, 0.25), Pos(0.25, 0.75)),
            Segment(Pos(0.14, 0.45), Pos(0., 0.65)),
            Segment(Pos(0.25, 0.75), Pos(0., 0.8)),
            Segment(Pos(0.25, 0.75), Pos(0.66, 0.875)),
            Segment(Pos(0.355, 0.), Pos(0.525, 0.185)),
            Segment(Pos(0.25, 0.5), Pos(0.75, 0.215)),
            Segment(Pos(1., 0.25), Pos(0.435, 0.55)),
        ])
