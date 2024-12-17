from typing import Any, Callable, Mapping, Sequence, Tuple

import fedjax
import jax

ClientId = bytes
Grads = fedjax.Params


@fedjax.dataclass
class ServerState:
    """State of server passed between rounds.

    Attributes:
      params: A pytree representing the server model parameters.
      opt_state: A pytree representing the server optimizer state.
    """

    params: fedjax.Params
    opt_state: fedjax.OptState


def federated_proximal(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey], Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
    mu: float,  # Proximal term coefficient
) -> fedjax.FederatedAlgorithm:
    """Builds the implementation of federated proximal."""

    def init(params: fedjax.Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]],
    ) -> Tuple[ServerState, Mapping[ClientId, Any]]:
        client_diagnostics = {}
        client_delta_params_weights = []
        for client_id, client_dataset, client_rng in clients:
            delta_params = client_update(
                server_state.params, client_dataset, client_rng
            )
            client_delta_params_weights.append((delta_params, len(client_dataset)))
            client_diagnostics[client_id] = {
                "delta_l2_norm": fedjax.tree_util.tree_l2_norm(delta_params)
            }
        mean_delta_params = fedjax.tree_util.tree_mean(client_delta_params_weights)
        server_state = server_update(server_state, mean_delta_params)
        return server_state, client_diagnostics

    def client_update(server_params, client_dataset, client_rng):
        """Performs client update with proximal regularization."""
        params = server_params
        opt_state = client_optimizer.init(params)
        for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
            client_rng, use_rng = jax.random.split(client_rng)
            grads = grad_fn(params, batch, use_rng)
            # Apply proximal term: adjust gradients based on the proximity to the global model.
            prox_term = jax.tree_util.tree_map(
                lambda p, sp: mu * (p - sp), params, server_params
            )
            grads = jax.tree_util.tree_map(lambda g, prox: g + prox, grads, prox_term)
            opt_state, params = client_optimizer.apply(grads, opt_state, params)
        delta_params = jax.tree_util.tree_map(lambda a, b: a - b, server_params, params)
        return delta_params

    def server_update(server_state, mean_delta_params):
        opt_state, params = server_optimizer.apply(
            mean_delta_params, server_state.opt_state, server_state.params
        )
        return ServerState(params, opt_state)

    return fedjax.FederatedAlgorithm(init, apply)
