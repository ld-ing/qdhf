from __future__ import annotations

import flax.struct
import jax
from jax import numpy as jnp
from kheperax.geoms import Pos, Segment


@flax.struct.dataclass
class Laser:
    pos: Pos
    angle: float
    range: float

    @jax.jit
    def get_segment(self) -> Segment:
        return Segment(
            p1=self.pos,
            p2=Pos(
                x=self.pos.x + self.range * jnp.cos(self.angle),
                y=self.pos.y + self.range * jnp.sin(self.angle),
            ),
        )

    @jax.jit
    def get_intersection_with_segment(self, segment: Segment) -> Pos:
        return self.get_segment().get_intersection_with(segment)

    def get_measure(self, array_segments, return_minus_one_if_out_of_range, std_noise, random_key):

        all_measures = jax.vmap(self.get_measure_for_segment)(array_segments)
        measure = jnp.min(all_measures)

        random_key, subkey = jax.random.split(random_key)
        noise = jax.random.normal(subkey) * std_noise
        measure = measure + noise
        measure = jnp.maximum(measure, 0.)

        if return_minus_one_if_out_of_range:
            out_of_range_value = -1.
        else:
            out_of_range_value = self.range

        measure = jnp.where(jnp.isinf(measure), out_of_range_value, measure)
        measure = jnp.where(measure > self.range, out_of_range_value, measure)

        return measure

    def get_measure_for_segment(self, segment):
        return self.pos.dist_to(self.get_intersection_with_segment(segment))
