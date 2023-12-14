import jax
import jax.numpy as jnp

from kheperax.geoms import Segment, Pos


class RenderingTools:
    @classmethod
    def place_circle(cls, array, center, radius, value):
        x, y = jnp.meshgrid(jnp.linspace(0, 1, array.shape[0]), jnp.linspace(0, 1, array.shape[1]))

        return jnp.where(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2,
            value,
            array
        )

    @classmethod
    def place_triangle(cls, array, point_1, point_2, point_3, value):
        x, y = jnp.meshgrid(jnp.linspace(0, 1, array.shape[0]), jnp.linspace(0, 1, array.shape[1]))

        return jnp.where(
            jnp.logical_and((x - point_1[0]) * (y - point_2[1]) - (x - point_2[0]) * (y - point_1[1]) <= 0,
                            jnp.logical_and(
                                (x - point_2[0]) * (y - point_3[1]) - (x - point_3[0]) * (y - point_2[1]) <= 0,
                                (x - point_3[0]) * (y - point_1[1]) - (x - point_1[0]) * (y - point_3[1]) <= 0)
                            ),
            value,
            array
        )

    @classmethod
    def place_rectangle(cls, array, start, width, height, value):
        x, y = jnp.meshgrid(jnp.linspace(0, 1, array.shape[0]), jnp.linspace(0, 1, array.shape[1]))

        return jnp.where(
            (x >= start[0] - width / 2) & (x <= start[0] + width / 2) & (y >= start[1]) & (y <= start[1] + height),
            value,
            array
        )

    @classmethod
    def get_distance_point_to_segment(cls, point: Pos, segment: Segment):
        return Pos.calculate_projection_on_segment(point, segment).dist_to(point)

    @classmethod
    def place_segments(cls, image, segments, value):
        x, y = jnp.meshgrid(jnp.linspace(0, 1, image.shape[0]), jnp.linspace(0, 1, image.shape[1]))

        matrix = jnp.concatenate([jnp.expand_dims(x, axis=-1), jnp.expand_dims(y, axis=-1)], axis=-1)

        get_distance_point_to_segment_v = jax.vmap(cls.get_distance_point_to_segment, in_axes=(0, None))
        get_distance_point_to_segment_vv = jax.vmap(get_distance_point_to_segment_v, in_axes=(0, None))
        get_distance_point_to_segment_vvv = jax.vmap(get_distance_point_to_segment_vv, in_axes=(None, 0))

        points = Pos(x=x, y=y)
        distances = get_distance_point_to_segment_vvv(points, segments)
        distances = jnp.min(distances, axis=(0,))
        print("distances", distances.shape)

        return jnp.where(
            distances <= 0.005,
            value,
            image
        )
