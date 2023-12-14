import jax


def get_batch_size(tree):
    batch_size = jax.tree_leaves(tree)[0].shape[0]
    return batch_size


def get_index_pytree(tree, index):
    return jax.tree_map(lambda x: x[index], tree)
