import pathlib
import torch
import json
import os

with open("config.json") as f:
    cfg = json.load(f)

os.environ["DATAPATH"] = cfg["rnastructuredata_path"]

if cfg["device"] is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(cfg["device"])

cpus = cfg["cpus"]
max_length = cfg["max_length"]
rnastructureexe_path = cfg["rnastructureexe_path"]
max_epochs = cfg["max_epochs"]
early_stopping_patience = cfg["early_stopping_patience"]
batch_size = cfg["batch_size"]
