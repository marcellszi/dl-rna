#!/usr/bin/env python3
import demomodel.network as network
import demomodel.config as config
import demomodel.utils as utils
import demomodel.data as data

import argparse
import pathlib
import torch
import sys


def get_args():
    parser = argparse.ArgumentParser(description="train the demonstrative model")
    parser.add_argument("train_path", help="path to training CT files")
    parser.add_argument("valid_path", help="path to validation CT files")
    parser.add_argument("output_path", help="path where model will be output")
    parser.add_argument(
        "--test_path",
        required=False,
        help="path to testing CT files, optional but will output statistics on test set",
    )
    parser.add_argument(
        "--grid-search", action="store_true", help="perform grid-search"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # output paths
    output_path = pathlib.Path(args.output_path)
    shape_path = output_path / "shape"
    grid_path = output_path / "grid-search"

    print("Creating directories...")
    output_path.mkdir(parents=True, exist_ok=True)
    shape_path.mkdir(parents=True, exist_ok=True)
    grid_path.mkdir(parents=True, exist_ok=True)
    print("Finished.")

    train_dataloader = torch.utils.data.DataLoader(
        data.CTData(args.train_path), batch_size=config.batch_size, pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        data.CTData(args.valid_path), batch_size=config.batch_size, pin_memory=True
    )
    if args.test_path is not None:
        test_dataloader = torch.utils.data.DataLoader(
            data.CTData(args.test_path), batch_size=config.batch_size, pin_memory=True
        )
    else:
        test_dataloader = None

    print("Building network...")
    model = network.DemonstrativeModel()
    model.to(config.device)
    print("Finished.")
    print("Training...")
    network.fit(
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        save_path=output_path,
        max_epochs=config.max_epochs,
        patience=config.early_stopping_patience,
    )
    print("Finished.")

    print("Starting grid-search...")
    print("Predicting validation shadows...")
    utils.predict_shape(model, valid_dataloader, shape_path)
    if args.grid_search:
        print("Folding grid...")
        utils.grid_search(args.valid_path, shape_path, grid_path)
        print("Grid folded.")
        print("Scoring grid...")
        utils.grid_score(args.valid_path, grid_path)
    print("Finished.")
