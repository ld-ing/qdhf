from __future__ import annotations

import flax.struct
import jax
from jax import numpy as jnp


@flax.struct.dataclass
class Posture:
    x: jnp.ndarray
    y: jnp.ndarray
    angle: jnp.ndarray

    def dist_to(self, other: Posture) -> jnp.ndarray:
        return self.dist_to_xy(other.x, other.y)

    def dist_to_xy(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(jnp.asarray([self.x, self.y]) - jnp.asarray([x, y]))

    def rotate(self, angle: angle) -> Posture:
        x_ = self.x * jnp.cos(angle) - self.y * jnp.sin(angle)
        y_ = self.x * jnp.sin(angle) + self.y * jnp.cos(angle)
        theta_ = self.normalize_angle(self.angle + angle)
        return Posture(x_, y_, theta_)

    def add_pos(self, p: Posture):
        return self.add_pos_xytheta(p.x, p.y, p.angle)

    def add_pos_xytheta(self,
                        x: jnp.ndarray,
                        y: jnp.ndarray,
                        theta: jnp.ndarray,
                        ) -> Posture:
        return Posture(self.x + x,
                       self.y + y,
                       self.normalize_angle(self.angle + theta),
                       )

    @staticmethod
    def normalize_angle(angle: jnp.ndarray) -> jnp.ndarray:
        return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi

    def move(self, d_l: jnp.ndarray, d_r: jnp.ndarray, wheels_dist: jnp.ndarray):
        old_pos = self
        alpha = (d_r - d_l) / wheels_dist

        def _if_alpha_high(_):
            r = (d_l / alpha) + (wheels_dist / 2)
            d_x = (jnp.cos(alpha) - 1) * r
            d_y = jnp.sin(alpha) * r
            delta_p = Posture(d_x, d_y, alpha)
            delta_p = delta_p.rotate(old_pos.angle - jnp.pi / 2)
            delta_p = delta_p.replace(angle=self.normalize_angle(alpha))
            return delta_p

        def _if_alpha_low(_):
            delta_p = Posture(
                x=d_l * jnp.cos(old_pos.angle),
                y=d_l * jnp.sin(old_pos.angle),
                angle=0.,
            )
            return delta_p

        delta_p = jax.lax.cond(jnp.abs(alpha) > 1e-10,
                               _if_alpha_high,
                               _if_alpha_low,
                               None)

        return self.add_pos(delta_p)
