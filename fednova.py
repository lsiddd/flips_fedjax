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


def federated_nova(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey], Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
) -> fedjax.FederatedAlgorithm:
    """Builds the implementation of federated normalized averaging (FedNova)."""

    def init(params: fedjax.Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]],
    ) -> Tuple[ServerState, Mapping[ClientId, Any]]:
        client_diagnostics = {}
        client_delta_params_weights = []
        total_steps_normalization = 0  # Keeps track of the sum of normalized steps

        for client_id, client_dataset, client_rng in clients:
            delta_params, num_steps = client_update(
                server_state.params, client_dataset, client_rng
            )
            client_delta_params_weights.append(
                (delta_params, len(client_dataset) * num_steps)
            )
            total_steps_normalization += num_steps
            client_diagnostics[client_id] = {
                "delta_l2_norm": fedjax.tree_util.tree_l2_norm(delta_params),
                "num_steps": num_steps,
            }

        # Normalize deltas using the total number of steps
        normalized_delta_params = fedjax.tree_util.tree_mean(
            client_delta_params_weights
        )
        normalized_delta_params = jax.tree_util.tree_map(
            lambda delta: delta / total_steps_normalization, normalized_delta_params
        )

        server_state = server_update(server_state, normalized_delta_params)
        return server_state, client_diagnostics

    def client_update(server_params, client_dataset, client_rng):
        """Performs client update and tracks the number of local steps."""
        params = server_params
        opt_state = client_optimizer.init(params)
        num_steps = 0

        for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
            num_steps += 1
            client_rng, use_rng = jax.random.split(client_rng)
            grads = grad_fn(params, batch, use_rng)
            opt_state, params = client_optimizer.apply(grads, opt_state, params)

        delta_params = jax.tree_util.tree_map(lambda a, b: a - b, server_params, params)
        scaling_factor = 1.0 / num_steps  # Normalize the delta parameters
        normalized_delta_params = jax.tree_util.tree_map(
            lambda delta: delta * scaling_factor, delta_params
        )

        return normalized_delta_params, num_steps

    def server_update(server_state, normalized_delta_params):
        """Updates the global model using normalized updates."""
        opt_state, params = server_optimizer.apply(
            normalized_delta_params, server_state.opt_state, server_state.params
        )
        return ServerState(params, opt_state)

    return fedjax.FederatedAlgorithm(init, apply)
