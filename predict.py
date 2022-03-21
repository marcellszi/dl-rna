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
    parser = argparse.ArgumentParser(description="predict using a demonstrative model")
    parser.add_argument("model_path", help="path to folder containing `model.pt`")
    parser.add_argument("seq_path", help="path to testing SEQ files")
    parser.add_argument(
        "output_path", help="path where CT and SHAPE files will be outout"
    )
    parser.add_argument(
        "-si",
        default=-0.6,
        type=float,
        help="intercept used with SHAPE restraints, default: -0.6 kcal/mol",
    )
    parser.add_argument(
        "-sm",
        default=1.8,
        type=float,
        help="slope used with SHAPE restraints, default: 1.8 kcal/mol",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # output paths
    output_path = pathlib.Path(args.output_path)
    shape_path = output_path / "shape"
    ct_path = output_path / "ct"

    print("Creating directories...")
    output_path.mkdir(parents=True, exist_ok=True)
    shape_path.mkdir(parents=True, exist_ok=True)
    ct_path.mkdir(parents=True, exist_ok=True)
    print("Finished.")

    test_dataloader = torch.utils.data.DataLoader(
        data.SEQData(args.seq_path), batch_size=config.batch_size, pin_memory=True
    )

    print("Loading network...")
    model = torch.load(output_path / "model.pt", map_location=config.device)
    model.to(config.device)
    print("Finished.")

    print("Predicting shadows...")
    utils.predict_shape(model, test_dataloader, shape_path)
    print("Finished.")
    print("Folding...")
    utils.fold(args.seq_path, shape_path, ct_path, si=args.si, sm=args.sm)
    print("Finished.")
