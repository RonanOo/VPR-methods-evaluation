import sys
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import models
import parser
import commons
import visualizations
from test_dataset import TestDataset


def test(
    positive_dist_threshold=None,
    method=None,
    backbone=None,
    descriptors_dimension=None,
    database_folder=None,
    queries_folder=None,
    num_workers=None,
    batch_size=None,
    exp_name=None,
    device=None,
    recall_values=None,
    num_preds_to_save=None,
    save_only_wrong_preds=None,
    model_file=None,
):
    start_time = datetime.now()
    output_folder = f"logs/{exp_name}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.setup_logging(output_folder, stdout="info")
    logging.info(
        f"Testing with {method} with a {backbone} backbone and descriptors dimension {descriptors_dimension}"
    )
    logging.info(f"The outputs are being saved in {output_folder}")

    model = models.get_model(method, backbone, descriptors_dimension, model_file)
    model = model.eval().to(device)

    test_ds = TestDataset(
        database_folder,
        queries_folder,
        positive_dist_threshold=positive_dist_threshold,
    )
    logging.info(f"Testing on {test_ds}")

    with torch.inference_mode():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        all_descriptors = np.empty(
            (len(test_ds), descriptors_dimension), dtype="float32"
        )
        for images, indices in database_dataloader:
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logging.debug(
            "Extracting queries descriptors for evaluation/testing using batch size 1"
        )
        queries_subset_ds = Subset(
            test_ds,
            list(
                range(test_ds.database_num, test_ds.database_num + test_ds.queries_num)
            ),
        )
        queries_dataloader = DataLoader(
            dataset=queries_subset_ds, num_workers=num_workers, batch_size=1
        )
        for images, indices in queries_dataloader:
            descriptors = model(images.to(device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.database_num :]
    database_descriptors = all_descriptors[: test_ds.database_num]

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(descriptors_dimension)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors

    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(recall_values))

    # For each query, check if the predictions are correct
    positives_per_query = test_ds.get_positives()
    recalls = np.zeros(len(recall_values))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(recall_values):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break

    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / test_ds.queries_num * 100
    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)]
    )
    logging.info(recalls_str)

    # Save visualizations of predictions
    if num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(
            predictions[:, :num_preds_to_save],
            test_ds,
            output_folder,
            save_only_wrong_preds,
        )
    return recalls_str
