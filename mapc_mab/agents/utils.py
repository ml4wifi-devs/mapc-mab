from functools import reduce
from itertools import chain, combinations, product

import jax
import jax.numpy as jnp
from jax.experimental import sparse


spstack = jax.jit(sparse.sparsify(jnp.stack))


def iter_tx(associations: dict) -> sparse.BCOO:
    """
    Iterate through all possible actions: combinations of active APs and associated stations.

    Parameters
    ----------
    associations : dict
        A dictionary mapping APs to a list of stations associated with each AP

    Returns
    -------
    sparse.BCOO
        A sparse matrix representing the TX matrix
    """

    aps = set(associations)
    stations = reduce(set.union, map(set, associations.values()), set())
    nodes = list(aps.union(stations))
    aps = list(map(nodes.index, aps))

    def _iter_edges():
        for active in chain.from_iterable(combinations(aps, r) for r in range(1, len(aps) + 1)):
            for stations in product(*((nodes.index(s) for s in associations[a]) for a in active)):
                yield tuple(zip(active, stations))

    spactions = [
        sparse.BCOO(args=(len(ac) * [1,], ac), shape=(len(nodes), len(nodes)),)
        for ac in _iter_edges()
    ]

    return spstack(spactions)
