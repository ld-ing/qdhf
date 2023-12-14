from __future__ import annotations

import flax.struct
import jax
import jax.numpy as jnp

from kheperax.posture import Posture


@flax.struct.dataclass
class Pos:
    x: float
    y: float

    def __sub__(self, other):
        return Pos(self.x - other.x, self.y - other.y)

    def dist_to(self, other: Pos) -> jnp.ndarray:
        return self.dist_to_xy(other.x, other.y)

    def dist_to_xy(self, x: float, y: float) -> jnp.ndarray:
        return jnp.linalg.norm(jnp.asarray([self.x, self.y]) - jnp.asarray([x, y]))

    def calculate_projection_on_segment(self, segment: Segment) -> Pos:
        x = self.x
        y = self.y
        x3 = segment.p1.x
        y3 = segment.p1.y
        x4 = segment.p2.x
        y4 = segment.p2.y
        u = ((x - x3) * (x4 - x3) + (y - y3) * (y4 - y3)) / ((x4 - x3) ** 2 + (y4 - y3) ** 2)
        u_clipped = jnp.clip(u, 0, 1)
        return Pos(x3 + u_clipped * (x4 - x3), y3 + u_clipped * (y4 - y3))

    @classmethod
    def from_posture(cls, posture: Posture):
        return cls(posture.x, posture.y)


@flax.struct.dataclass
class Disk:
    pos: Pos
    radius: float

    def does_intersect_segment(self, segment: Segment) -> bool:
        projection = self.pos.calculate_projection_on_segment(segment)
        return projection.dist_to(self.pos) <= self.radius

    def collides(self, array_segments):
        return jnp.any(jax.vmap(self.does_intersect_segment)(array_segments))


@flax.struct.dataclass
class Line:
    p1: Pos
    p2: Pos


@flax.struct.dataclass
class Segment:
    p1: Pos
    p2: Pos

    def check_intersection_with(self, other: Segment) -> bool:
        def ccw(a: Pos, b: Pos, c: Pos):
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

        return jnp.logical_and(
            jnp.logical_not(jnp.isclose(ccw(self.p1, other.p1, other.p2), ccw(self.p2, other.p1, other.p2))),
            jnp.logical_not(jnp.isclose(ccw(self.p1, self.p2, other.p1), ccw(self.p1, self.p2, other.p2)))
        )

    def get_intersection_with(self, other: Segment) -> Pos:
        x1, y1 = self.p1.x, self.p1.y
        x2, y2 = self.p2.x, self.p2.y
        x3, y3 = other.p1.x, other.p1.y
        x4, y4 = other.p2.x, other.p2.y

        def is_false(_):
            return Pos(jnp.inf, jnp.inf)

        def is_true(_):
            d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

            return Pos(
                ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d,
                ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d,
            )

        return jax.lax.cond(self.check_intersection_with(other), is_true, is_false, None)
