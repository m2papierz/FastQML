# Copyright (C) 2024 Mateusz Papierz
#
# This file is part of the FastQML library
#
# The FastQML library is free software released under the GNU General Public License v3
# or later. You can redistribute and/or modify it under the terms of the GPL v3.
# See the LICENSE file in the project root or <https://www.gnu.org/licenses/gpl-3.0.html>.
#
# THERE IS NO WARRANTY for the FastQML library, as per Section 15 of the GPL v3.

from collections import OrderedDict

from typing import Union
from typing import Dict
from typing import Any
from typing import List

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class EstimatorLayerParameters:
    """
    A class to hold parameters for an estimator layer.

    Attributes:
        q_params: Quantum parameters.
        c_params: Classical parameters.
        batch_stats: Batch statistics for classical model.
    """
    def __init__(
            self,
            q_params: Union[jnp.ndarray, None],
            c_params: Union[jnp.ndarray, Dict[str, Any], None],
            batch_stats: Union[jnp.ndarray, Dict[str, Any], None]
    ):
        self.q_params = q_params
        self.c_params = c_params
        self.batch_stats = batch_stats

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        children = []
        aux_data = {
            'q_params': self.q_params,
            'c_params': self.c_params,
            'batch_stats': self.batch_stats
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

    def get_params_num(self):
        """
        Initializes the total_params attribute based on the weights and batch stats.
        """
        total_params = 0
        if self.q_params is not None:
            total_params += len(self.q_params.ravel())
        if self.c_params is not None:
            total_params += sum(x.size for x in jax.tree_leaves(self.c_params))
        return total_params

    def __iter__(self):
        """
        Allow unpacking of the class instance.
        """
        yield self.q_params
        yield self.c_params
        yield self.batch_stats

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"    q_params={self.q_params!r},\n"
            f"    c_params={self.c_params!r},\n"
            f"    batch_stats={self.batch_stats!r},\n"
            f"    total_params={self.get_params_num()!r}\n)"
        )


@register_pytree_node_class
class EstimatorParameters:
    """
    A class to hold parameters of the estimator.

    Attributes:
        layers_params: List of the parameters of estimator layers.
    """
    def __init__(
            self,
            layers_params: List[EstimatorLayerParameters]
    ):
        self.layers_params = layers_params
        self.parameters = self._init_parameters()

    def _init_parameters(self) -> OrderedDict:
        q_counts, c_counts = 0, 0
        parameters = OrderedDict()

        for layer_params in self.layers_params:
            q_params, c_params, batch_stats = layer_params

            if q_params is not None:
                parameters[f"QuantumLayer{q_counts}"] = q_params
                q_counts += 1

            if c_params is not None:
                if batch_stats is not None:
                    parameters[f"ClassicalLayer{c_counts}"] = [c_params, batch_stats]
                else:
                    parameters[f"ClassicalLayer{c_counts}"] = c_params
                c_counts += 1

        return parameters

    def tree_flatten(self):
        """
        Prepares the class instance for JAX tree operations.
        """
        children = [self.parameters]
        aux_data = {'layers_params': self.layers_params}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from JAX tree operations.
        """
        return cls(*children, **aux_data)

    def __repr__(self):
        parameters_repr = ",\n    ".join(
            f"    {key!r}: {value!r}" for key, value in self.parameters.items())
        return (
            f"{self.__class__.__name__}(\n"
            f"    parameters=OrderedDict([\n    "
            f"{parameters_repr}\n"
            f"    ])\n"
            f")"
        )
