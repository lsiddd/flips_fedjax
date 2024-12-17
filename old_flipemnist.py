import fedjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import app


def main(_):
    # Load train and test federated data for EMNIST.
    train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False)

    # Create CNN model with dropout.
    model = fedjax.models.emnist.create_conv_model(only_digits=False)

    # Scalar loss function with model parameters, batch of examples, and seed PRNGKey as input.
    def loss(params, batch, rng):
        preds = model.apply_for_train(params, batch, rng)
        example_loss = model.train_loss(batch, preds)
        return jnp.mean(example_loss)

    # Gradient function of `loss` w.r.t. to model `params` (jitted for speed).
    grad_fn = jax.jit(jax.grad(loss))

    # Define SHAP-based importance scoring function
    def shap_fn(params, batch):
        """
        Compute SHAP importance scores for each layer of the model.
        Handle nested pytree structures in FedJAX model parameters.
        """
        importance_scores = {}

        def compute_layer_importance(layer_weights):
            # Recursively handle nested dictionaries
            if isinstance(layer_weights, dict):
                return sum(
                    compute_layer_importance(sub_layer)
                    for sub_layer in layer_weights.values()
                )
            else:
                # Compute mean absolute weight for the current layer
                return jnp.mean(jnp.abs(layer_weights))

        for layer_name, layer_weights in params.items():
            importance_scores[layer_name] = compute_layer_importance(layer_weights)

        return importance_scores

    # Pruning thresholds for each layer
    pruning_thresholds = {"conv1": 0.2, "conv2": 0.3, "dense1": 0.5, "dense2": 0.7}

    # Client selection weights for multifactor selection
    client_selection_weights = [0.4, 0.3, 0.2, 0.1]

    # Create FLIPS algorithm
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(
        learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4)
    )
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    from flips import flips_algorithm  # Import the FLIPS implementation

    algorithm = flips_algorithm(
        grad_fn=grad_fn,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        client_batch_hparams=client_batch_hparams,
        shap_fn=shap_fn,
        pruning_thresholds=pruning_thresholds,
        client_selection_weights=client_selection_weights,
    )

    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    # Train and eval loop.
    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
        fd=train_fd, num_clients=10, seed=0
    )

    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    rounds = []

    # Increase number of rounds for better results.
    for round_num in range(1, 50):
        clients = train_client_sampler.sample()
        server_state, client_diagnostics = algorithm.apply(server_state, clients)
        print(f"[round {round_num}]")

        # Periodically evaluate the trained server model parameters.
        if True:
            client_ids = [cid for cid, _, _ in clients]
            train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
            test_eval_datasets = [cds for _, cds in test_fd.get_clients(client_ids)]
            train_eval_batches = fedjax.padded_batch_client_datasets(
                train_eval_datasets, batch_size=256
            )
            test_eval_batches = fedjax.padded_batch_client_datasets(
                test_eval_datasets, batch_size=256
            )

            train_metrics = fedjax.evaluate_model(
                model, server_state.params, train_eval_batches
            )
            test_metrics = fedjax.evaluate_model(
                model, server_state.params, test_eval_batches
            )

            train_accuracy = train_metrics["accuracy"]
            train_loss = train_metrics["loss"]
            test_accuracy = test_metrics["accuracy"]
            test_loss = test_metrics["loss"]

            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            test_accuracies.append(test_accuracy)
            test_losses.append(test_loss)
            rounds.append(round_num)

            print(f"[round {round_num}] train_metrics={train_metrics}")
            print(f"[round {round_num}] test_metrics={test_metrics}")

    # Save final trained model parameters to file.
    fedjax.serialization.save_state(server_state.params, "/tmp/params")

    # Plot the training and testing metrics.
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
