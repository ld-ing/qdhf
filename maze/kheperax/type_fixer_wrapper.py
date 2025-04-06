import jax.numpy as jnp

from brax.envs import base as brax_env


class TypeFixerWrapper(brax_env.Wrapper):
    def reset(self, rng: jnp.ndarray):
        reset_state = self.env.reset(rng)
        reset_state.info["truncation"] = jnp.asarray(reset_state.info["truncation"], dtype=jnp.int32)

        return reset_state

    def step(self, state, action: jnp.ndarray):
        state = self.env.step(state, action)
        state.info["truncation"] = jnp.asarray(state.info["truncation"], dtype=jnp.int32)
        return state
