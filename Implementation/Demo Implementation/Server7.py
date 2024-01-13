import flwr as fl
import numpy as np
from pathlib import Path

def evaluate_metrics_aggregation_fn(results):
    total_num_samples = 0
    total_accuracy = 0.0
    count = 0
    for result in results:
        if len(result) == 2:
            num_samples, accuracy = result
            total_num_samples += num_samples
            accuracy_value = accuracy["Accuracy"]
            total_accuracy += accuracy_value 
            count += 1
        else:
            print("WARNING: Invalid result format")
    if count > 0:
        metrics_aggregated = {
            "num_samples": total_num_samples,
            "accuracy": total_accuracy / count,
        }
    else:
        metrics_aggregated = None

    if metrics_aggregated is None:
        return {}
    else:
        return metrics_aggregated


strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=7,
    min_evaluate_clients=7,
    min_available_clients=7,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)


fl.server.start_server(
  server_address="10.31.0.26:8082",
  strategy=strategy,
  config=fl.server.ServerConfig(num_rounds=50)
)
