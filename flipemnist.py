from typing import Any, Callable, Mapping, Sequence, Tuple

import fedjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import app

# import fedjax.dataclass as dataclass

ClientId = bytes
Grads = fedjax.Params


def compute_dynamic_client_weights(client_diagnostics, alpha=0.5, beta=0.5):
    """
    Compute dynamic weights for client selection based on client diagnostics.

    Args:
        client_diagnostics: A dictionary with client metrics like 'delta_l2_norm' and 'metrics'.
        alpha: Weight for gradient norm-based importance.
        beta: Weight for loss improvement-based importance.

    Returns:
        A normalized dictionary of client weights.
    """
    weights = {}
    for client_id, diagnostics in client_diagnostics.items():
        # Gradient norm as importance
        grad_norm = diagnostics.get("delta_l2_norm", 0.0)
        
        # Custom metrics (like loss reduction or other values)
        metrics = diagnostics.get("metrics", [0.0])
        loss_reduction = metrics[0]  # Assume the first metric represents loss improvement
        
        # Combine gradient norm and loss reduction with alpha, beta
        combined_score = alpha * grad_norm + beta * loss_reduction
        weights[client_id] = max(combined_score, 0.0)  # Ensure non-negative weights
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        raise ValueError("All computed weights are zero. Check client diagnostics.")
    return weights


def multifactor_client_selection_with_dynamic_weights(client_diagnostics):
    """
    Select clients dynamically based on computed weights.

    Args:
        client_diagnostics: A dictionary of client diagnostics.

    Returns:
        A sorted list of selected clients with their weights.
    """
    dynamic_weights = compute_dynamic_client_weights(client_diagnostics)
    return sorted(dynamic_weights.items(), key=lambda x: x[1], reverse=True)


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

# Modify the flips_algorithm function
def flips_algorithm(
    grad_fn: Callable[[fedjax.Params, fedjax.BatchExample, fedjax.PRNGKey], Grads],
    client_optimizer: fedjax.Optimizer,
    server_optimizer: fedjax.Optimizer,
    client_batch_hparams: fedjax.ShuffleRepeatBatchHParams,
    shap_fn: Callable,
    client_selection_weights: list,
) -> fedjax.FederatedAlgorithm:
    """FLIPS Algorithm."""

    def init(params: fedjax.Params) -> ServerState:
        opt_state = server_optimizer.init(params)
        return ServerState(params, opt_state)

    def apply(
        server_state: ServerState,
        clients: Sequence[Tuple[ClientId, fedjax.ClientDataset, fedjax.PRNGKey]],
        pruning_thresholds: dict,  # Dynamically passed thresholds
    ) -> Tuple[ServerState, Mapping[ClientId, Any]]:
        client_updates = {}
        client_diagnostics = {}

        for client_id, client_dataset, client_rng in clients:
            client_update, diagnostics = client_update_fn(
                server_state.params, client_dataset, client_rng, pruning_thresholds
            )
            client_updates[client_id] = client_update
            client_diagnostics[client_id] = diagnostics

            # Replace static client selection weights
            selected_clients = multifactor_client_selection_with_dynamic_weights(client_diagnostics)

            # Extract updates for selected clients based on weights
            selected_updates = {
                client_id: client_updates[client_id]
                for client_id, _ in selected_clients
            }

            # Compute the normalized dynamic weights
            dynamic_weights = {client_id: weight for client_id, weight in selected_clients}

            # SHAP-based importance weighting
            flat_importance_scores = flatten_importance_scores(
                {c: d["importance"] for c, d in client_diagnostics.items()}
            )
            aggregated_update = importance_weighted_aggregation(selected_updates, dynamic_weights)


        # Server update
        server_state = server_update(server_state, aggregated_update)
        return server_state, client_diagnostics

    def client_update_fn(server_params, client_dataset, client_rng, pruning_thresholds):
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
        """Compute metrics for client selection."""
        return [1.0, 2.0]  # Replace with actual metric calculations

    return fedjax.FederatedAlgorithm(init, apply)

def main(_):
    # Load train and test federated data for EMNIST.
    train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False)

    # Create CNN model with dropout.
    model = fedjax.models.emnist.create_conv_model(only_digits=False)

    # Loss function for training.
    def loss(params, batch, rng):
        preds = model.apply_for_train(params, batch, rng)
        example_loss = model.train_loss(batch, preds)
        return jnp.mean(example_loss)

    # Gradient function.
    grad_fn = jax.jit(jax.grad(loss))

    # SHAP-based importance scoring function.
    def shap_fn(params, batch):
        """
        Compute SHAP importance scores for each layer of the model.
        Handle nested pytree structures in FedJAX model parameters.
        """
        importance_scores = {}

        def compute_layer_importance(layer_weights):
            if isinstance(layer_weights, dict):
                return sum(
                    compute_layer_importance(sub_layer)
                    for sub_layer in layer_weights.values()
                )
            return jnp.mean(jnp.abs(layer_weights))

        for layer_name, layer_weights in params.items():
            importance_scores[layer_name] = compute_layer_importance(layer_weights)

        return importance_scores

    # Dynamic pruning threshold calculation.
    def calculate_pruning_thresholds(importance_scores, prune_quantile=0.5):
        """
        Calculate dynamic pruning thresholds based on SHAP scores.
        prune_quantile determines the percentage of least important weights to prune.
        """
        thresholds = {}
        for layer, score in importance_scores.items():
            threshold = jnp.quantile(jnp.array(score), prune_quantile)
            thresholds[layer] = threshold
        return thresholds

    # Client selection weights
    client_selection_weights = [0.4, 0.3, 0.2, 0.1]

    # FLIPS algorithm setup
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(
        learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4)
    )
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    # Initialize FLIPS algorithm
    algorithm = flips_algorithm(
        grad_fn=grad_fn,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        client_batch_hparams=client_batch_hparams,
        shap_fn=shap_fn,
        client_selection_weights=client_selection_weights,
    )

    # Initialize model and server state.
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    # Training and evaluation loop.
    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
        fd=train_fd, num_clients=10, seed=0
    )

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    rounds = []

    for round_num in range(1, 50):  # Adjust the range for training rounds.
        # Sample clients for this round
        clients = train_client_sampler.sample()

        # Compute SHAP scores for dynamic pruning thresholds
        client_batch = next(iter(clients[0][1].shuffle_repeat_batch(client_batch_hparams)))
        importance_scores = shap_fn(server_state.params, client_batch)
        pruning_thresholds = calculate_pruning_thresholds(
            importance_scores, prune_quantile=0.5
        )

        # Apply FLIPS algorithm with updated thresholds
        server_state, client_diagnostics = algorithm.apply(
            server_state, clients, pruning_thresholds
        )

        print(f"[Round {round_num}] Dynamic Pruning Thresholds: {pruning_thresholds}")

        # Periodic evaluation
        client_ids = [cid for cid, _, _ in clients]
        train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
        test_eval_datasets = [cds for _, cds in test_fd.get_clients(client_ids)]
        train_eval_batches = fedjax.padded_batch_client_datasets(
            train_eval_datasets, batch_size=256
        )
        test_eval_batches = fedjax.padded_batch_client_datasets(
            test_eval_datasets, batch_size=256
        )

        # Evaluate training metrics
        train_metrics = fedjax.evaluate_model(
            model, server_state.params, train_eval_batches
        )
        test_metrics = fedjax.evaluate_model(
            model, server_state.params, test_eval_batches
        )

        # Collect metrics
        train_accuracy = train_metrics["accuracy"]
        train_loss = train_metrics["loss"]
        test_accuracy = test_metrics["accuracy"]
        test_loss = test_metrics["loss"]

        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss)
        rounds.append(round_num)

        # Print metrics
        print(f"[Round {round_num}] Train Metrics: {train_metrics}")
        print(f"[Round {round_num}] Test Metrics: {test_metrics}")

    # Save final trained model parameters to file.
    fedjax.serialization.save_state(server_state.params, "/tmp/params")

    # Plot metrics.
    plt.figure()
    plt.plot(rounds, train_accuracies, label="Train Accuracy")
    plt.plot(rounds, test_accuracies, label="Test Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Rounds")
    plt.legend()
    plt.grid()
    plt.savefig("accuracy.png")

    plt.figure()
    plt.plot(rounds, train_losses, label="Train Loss")
    plt.plot(rounds, test_losses, label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss over Rounds")
    plt.legend()
    plt.grid()
    plt.savefig("loss.png")


if __name__ == "__main__":
    app.run(main)
