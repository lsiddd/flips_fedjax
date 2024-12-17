from typing import Any, Callable, Mapping, Sequence, Tuple

import fedjax
import jax

# import fedjax.dataclass as dataclass

ClientId = bytes
Grads = fedjax.Params


@fedjax.dataclass
class ServerState:
    """State of server passed between rounds."""

    params: fedjax.Params
    opt_state: fedjax.OptState


def shap_importance_scores(params, batch, shap_fn):
    """Compute SHAP importance scores for model layers."""
    return shap_fn(params, batch)


def selective_layer_pruning(params, importance_scores, pruning_thresholds):
    """Prune model layers selectively based on importance scores."""
    pruned_params = {}
    for layer, weights in params.items():
        threshold = pruning_thresholds.get(layer, 0.0)
        importance = importance_scores.get(layer, 0.0)
        if importance < threshold:
            pruned_params[layer] = jax.tree_util.tree_map(lambda w: 0.0, weights)
        else:
            pruned_params[layer] = weights
    return pruned_params


def flatten_importance_scores(importance_scores):
    """Flatten importance scores by summing layer scores for each client."""
    flattened_scores = {}
    for client_id, layer_scores in importance_scores.items():
        if isinstance(layer_scores, dict):  # If layer_scores is a dictionary
            total_score = sum(float(score) for score in layer_scores.values())
        else:  # If layer_scores is a JAX array or similar structure
            total_score = float(jax.numpy.sum(layer_scores))
        flattened_scores[client_id] = total_score
    return flattened_scores


def importance_weighted_aggregation(client_updates, importance_scores):
    """Aggregate client updates using importance scores."""
    # Flatten importance scores to get a single value per client
    flat_importance_scores = flatten_importance_scores(importance_scores)

    # Compute total weight
    total_weight = sum(flat_importance_scores.values())
    if total_weight == 0:
        raise ValueError("Total importance weight is zero. Check importance_scores.")

    # Initialize aggregated update with zeros
    aggregated_update = jax.tree_util.tree_map(
        lambda x: jax.numpy.zeros_like(x), next(iter(client_updates.values()))
    )

    # Aggregate updates using normalized importance scores
    for client_id, update in client_updates.items():
        client_weight = flat_importance_scores.get(client_id, 0.0) / total_weight

        # Apply the weighted update to the aggregated result
        aggregated_update = jax.tree_util.tree_map(
            lambda a, b: a + client_weight * b, aggregated_update, update
        )

    return aggregated_update


def multifactor_client_selection(client_metrics, weights):
    """Select clients based on multifactor scores."""
    scores = {
        client_id: sum(metric * weight for metric, weight in zip(metrics, weights))
        for client_id, metrics in client_metrics.items()
    }
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[: len(weights)]


def flips_algorithm(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey], Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
    shap_fn: Callable,
    pruning_thresholds: dict,
    client_selection_weights: list,
) -> fedjax.FederatedAlgorithm:
    """FLIPS Algorithm."""

    def init(params: fedjax.Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]],
    ) -> Tuple[ServerState, Mapping[ClientId, Any]]:
        client_updates = {}
        client_diagnostics = {}

        for client_id, client_dataset, client_rng in clients:
            client_update, diagnostics = client_update_fn(
                server_state.params, client_dataset, client_rng
            )
            client_updates[client_id] = client_update
            client_diagnostics[client_id] = diagnostics

        # Client selection
        selected_clients = multifactor_client_selection(
            {
                client_id: diagnostics["metrics"]
                for client_id, diagnostics in client_diagnostics.items()
            },
            client_selection_weights,
        )
        selected_updates = {
            client_id: client_updates[client_id] for client_id, _ in selected_clients
        }

        # SHAP-based importance weighting
        # Before calling importance_weighted_aggregation
        flat_importance_scores = flatten_importance_scores(
            {c: d["importance"] for c, d in client_diagnostics.items()}
        )
        aggregated_update = importance_weighted_aggregation(
            selected_updates, flat_importance_scores
        )

        # Server update
        server_state = server_update(server_state, aggregated_update)
        return server_state, client_diagnostics

    def client_update_fn(server_params, client_dataset, client_rng):
        """Perform client update."""
        params = server_params
        opt_state = client_optimizer.init(params)
        for batch in client_dataset.shuffle_repeat_batch(client_batch_hparams):
            client_rng, use_rng = jax.random.split(client_rng)
            grads = grad_fn(params, batch, use_rng)
            opt_state, params = client_optimizer.apply(grads, opt_state, params)

        importance_scores = shap_importance_scores(params, batch, shap_fn)
        pruned_params = selective_layer_pruning(
            params, importance_scores, pruning_thresholds
        )

        delta_params = jax.tree_util.tree_map(
            lambda a, b: a - b, server_params, pruned_params
        )
        diagnostics = {
            "delta_l2_norm": fedjax.tree_util.tree_l2_norm(delta_params),
            "importance": importance_scores,
            "metrics": compute_client_metrics(pruned_params, client_dataset),
        }
        return delta_params, diagnostics

    def server_update(server_state, aggregated_update):
        opt_state, params = server_optimizer.apply(
            aggregated_update, server_state.opt_state, server_state.params
        )
        return ServerState(params, opt_state)

    def compute_client_metrics(params, dataset):
        """Compute metrics for client selection (e.g., accuracy, communication cost)."""
        # Placeholder implementation
        return [1.0, 2.0]  # Replace with actual metric calculations

    return fedjax.FederatedAlgorithm(init, apply)
