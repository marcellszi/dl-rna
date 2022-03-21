#!/usr/bin/env python3
import demomodel.network as network
import demomodel.config as config
import demomodel.utils as utils
import demomodel.data as data

import argparse
import pathlib
import torch
import csv


def get_args():
    parser = argparse.ArgumentParser(
        description="calculate PPV, sensitivity, and F1 for CT files"
    )
    parser.add_argument("pred_path", help="path to predicted CT files")
    parser.add_argument("true_path", help="path to ground-truth CT files")
    parser.add_argument("output_path", help="path where CSV will be output")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    pred_path = pathlib.Path(args.pred_path)
    true_path = pathlib.Path(args.true_path)

    with open(args.output_path, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["name", "ppv", "sen", "f1"])
        for f in pred_path.glob("*.ct"):
            _, _, pred_pairings = utils.read_ct(f)
            _, _, true_pairings = utils.read_ct(true_path / f.name)

            ppv = utils.score_ppv(true_pairings, pred_pairings)
            sen = utils.score_sen(true_pairings, pred_pairings)
            f1 = utils.score_f1(true_pairings, pred_pairings)

            csvwriter.writerow([f.stem, ppv, sen, f1])
            print(f.stem, ppv, sen, f1)
