import fedjax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from absl import app, flags

import fed_avg
import fed_prox
import fednova

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "algorithm", "fedprox", "The algorithm to use: fednova, fedprox, fedavg, or all."
)

ALGORITHMS = ["fedprox", "fednova", "fedavg"]


def get_algorithm(
    name, grad_fn, client_optimizer, server_optimizer, client_batch_hparams
):
    if name == "fednova":
        return fednova.federated_nova(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams
        )
    elif name == "fedprox":
        return fed_prox.federated_proximal(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams, mu=0.1
        )
    elif name == "fedavg":
        return fed_avg.federated_averaging(
            grad_fn, client_optimizer, server_optimizer, client_batch_hparams
        )
    else:
        raise ValueError(f"Unsupported algorithm: {name}")


def train_and_evaluate(
    algorithm_name,
    model,
    grad_fn,
    client_optimizer,
    server_optimizer,
    client_batch_hparams,
    train_fd,
    test_fd,
):
    algorithm = get_algorithm(
        algorithm_name,
        grad_fn,
        client_optimizer,
        server_optimizer,
        client_batch_hparams,
    )

    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    # Train and eval loop.
    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
        fd=train_fd, num_clients=10, seed=0
    )

    train_accuracies, test_accuracies, train_losses, test_losses, rounds = (
        [],
        [],
        [],
        [],
        [],
    )

    for round_num in range(1, 50):
        clients = train_client_sampler.sample()
        server_state, client_diagnostics = algorithm.apply(server_state, clients)

        if True:  # Evaluate every round
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

            train_accuracies.append(train_metrics["accuracy"])
            train_losses.append(train_metrics["loss"])
            test_accuracies.append(test_metrics["accuracy"])
            test_losses.append(test_metrics["loss"])
            rounds.append(round_num)

            print(f"[round {round_num}] train_metrics={train_metrics}")
            print(f"[round {round_num}] test_metrics={test_metrics}")

    # Save plots
    plt.figure()
    plt.plot(rounds, train_accuracies, label="Train Accuracy")
    plt.plot(rounds, test_accuracies, label="Test Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over Rounds - {algorithm_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{algorithm_name}_accuracy.png")

    plt.figure()
    plt.plot(rounds, train_losses, label="Train Loss")
    plt.plot(rounds, test_losses, label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Loss over Rounds - {algorithm_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{algorithm_name}_loss.png")

    return rounds, test_accuracies


def main(_):
    algorithm = FLAGS.algorithm

    # Load train and test federated data for EMNIST.
    train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False)

    # Create CNN model with dropout.
    model = fedjax.models.emnist.create_conv_model(only_digits=False)

    def loss(params, batch, rng):
        preds = model.apply_for_train(params, batch, rng)
        example_loss = model.train_loss(batch, preds)
        return jnp.mean(example_loss)

    grad_fn = jax.jit(jax.grad(loss))
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(
        learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4)
    )
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)

    if algorithm == "all":
        results = {}
        for alg in ALGORITHMS:
            rounds, accuracies = train_and_evaluate(
                alg,
                model,
                grad_fn,
                client_optimizer,
                server_optimizer,
                client_batch_hparams,
                train_fd,
                test_fd,
            )
            results[alg] = (rounds, accuracies)

        # Plot accuracy comparison
        plt.figure()
        for alg, (rounds, accuracies) in results.items():
            plt.plot(rounds, accuracies, label=alg)
        plt.xlabel("Round")
        plt.ylabel("Test Accuracy")
        plt.title("Accuracy Comparison Across Algorithms")
        plt.legend()
        plt.grid()
        plt.savefig("accuracy_comparison.png")

    else:
        train_and_evaluate(
            algorithm,
            model,
            grad_fn,
            client_optimizer,
            server_optimizer,
            client_batch_hparams,
            train_fd,
            test_fd,
        )


if __name__ == "__main__":
    app.run(main)
