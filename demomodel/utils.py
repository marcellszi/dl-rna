import demomodel.config as config

import sklearn.metrics
import multiprocessing
import numpy as np
import subprocess
import itertools
import tempfile
import pathlib
import torch
import json
import tqdm
import sys


def read_ct(path, number=0):
    bases = []
    pairings = []
    with open(path) as f:
        # deal w/ header
        header = f.readline()
        if header[0] == ">":
            length = int(header.split()[2])
        else:
            length = int(header.split()[0])

        title = " ".join(header.split()[1:])
        f.seek(0)

        # skip to structure corresponding to number
        start_index = number * length + number
        for _ in range(start_index):
            next(f)

        for i, line in enumerate(f):
            # deal w/ header for nth structure
            if i == 0:
                if header[0] == ">":
                    length = int(line.split()[2])
                else:
                    length = int(header.split()[0])

                title = " ".join(line.split()[1:])
                continue

            bn, b, _, _, p, _ = line.split()

            if int(bn) != i:
                raise NotImplementedError(
                    "Skipping indices in CT files is not supported."
                )

            bases.append(b)
            pairings.append(int(p) - 1)

            if i == length:
                break

    # this shouldn't really ever happen, probs unnecessary check
    if length != len(bases) and length != len(pairings):
        raise RuntimeError("Length of parsed RNA does not match expected length.")

    return title, bases, pairings


def read_seq(path):
    title = None
    sequence = ""
    with open(path) as f:
        for line in f:
            if line[0] == ";":
                continue
            elif title is None:
                title = line.rstrip()
            else:
                sequence += "".join(line.split())

    return title, sequence[:-1]


def roc_auc(y_pred, y_true):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    try:
        auc = sklearn.metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.5

    return auc


def _fold(seq_file, ct_file, shape_file=None, si=-0.6, sm=1.8):
    cmd = [
        pathlib.Path(config.rnastructureexe_path) / "Fold",
        str(seq_file),
        str(ct_file),
    ]

    if shape_file is not None:
        cmd += ["-sh", str(shape_file), "-si", f"{si:.2f}", "-sm", f"{sm:.2f}"]

    sp = subprocess.run(cmd, capture_output=True)
    return sp


def write_shape(s, output_path, length=None):
    if length is None:
        length = len(s)
    with open(output_path, "w") as f:
        for k in range(length):
            f.write(f"{k+1}\t{(1-s[k]):.3f}\n")


def predict_shape(model, dataloader, output_path):
    with torch.no_grad():
        for i, (metadata, x, _) in enumerate(dataloader):
            x, _ = x.to(config.device), None
            y_pred = model(x)
            for j, s in enumerate(y_pred):
                name = metadata["name"][j]
                length = metadata["length"][j]
                write_shape(s, output_path / f"{name}.shape", length=length)


def _fold_grid_datum(grid_point):
    fps, b, m = grid_point
    ct_path, save_path, shape_path = fps
    seq_f = tempfile.NamedTemporaryFile()
    ct2seq(ct_path, seq_f.name)
    _fold(seq_f.name, save_path, shape_path, si=b, sm=m)
    seq_f.close()


def ct2seq(ct_path, seq_path):
    title, bases, pairings = read_ct(ct_path)

    with open(seq_path, "w") as f:
        f.write(f";\n{title}\n{''.join(bases)}1")


def grid_search(valid_path, shape_path, save_path, grid_size=11):
    files = list(pathlib.Path(valid_path).glob("*.ct"))

    bl, bu = 0.0, -3.0
    ml, mu = 0.0, 5.0
    b = np.linspace(bl, bu, num=grid_size)
    m = np.linspace(ml, mu, num=grid_size)

    bm = list(itertools.product(b, m))
    grid_names = {k: i for i, k in enumerate(bm)}
    for i, _ in enumerate(bm):
        grid_path = save_path / str(i)
        grid_path.mkdir(exist_ok=True)
    with open(save_path / "grid-idx.json", "w") as f:
        json.dump({i: k for i, k in enumerate(bm)}, f)

    grid = []
    for f, b_, m_ in itertools.product(files, b, m):
        fps = (
            f,
            save_path / str(grid_names[(b_, m_)]) / f"{f.stem}.ct",
            shape_path / f"{f.stem}.shape",
        )
        grid.append((fps, b_, m_))

    with multiprocessing.Pool(config.cpus) as p:
        list(
            tqdm.tqdm(
                p.imap_unordered(_fold_grid_datum, grid),
                total=grid_size ** 2 * len(files),
            )
        )


def score_ppv(correct, test):
    assert len(test) == len(correct)

    basepairs = 0
    score = 0

    for i in range(len(correct)):
        if test[i] > i:
            basepairs += 1

    if basepairs == 0:
        return np.nan

    for i in range(len(test)):
        if test[i] > i:
            if test[i] == correct[i]:
                score += 1
            elif test[i] == correct[i] + 1:
                score += 1
            elif test[i] == correct[i] - 1:
                score += 1
            elif test[i] == correct[i + 1]:
                score += 1
            elif test[i] == correct[i - 1]:
                score += 1
    return score / basepairs


def score_sen(correct, test):
    assert len(test) == len(correct)

    basepairs = 0
    score = 0

    for i in range(len(correct)):
        if correct[i] > i:
            basepairs += 1

    if basepairs == 0:
        return np.nan

    for i in range(len(test)):
        if correct[i] > i:
            if test[i] == correct[i]:
                score += 1
            elif test[i] + 1 == correct[i]:
                score += 1
            elif test[i] - 1 == correct[i]:
                score += 1
            elif test[i + 1] == correct[i]:
                score += 1
            elif test[i - 1] == correct[i]:
                score += 1
    return score / basepairs


def score_f1(correct, test):
    ppv, sen = score_ppv(correct, test), score_sen(correct, test)
    return 2 * (ppv * sen) / (ppv + sen + sys.float_info.epsilon)


def grid_score(valid_path, grid_path, grid_size=11):
    files = pathlib.Path(valid_path).glob("*.ct")

    with open(grid_path / "grid-idx.json") as f:
        d = json.load(f)

    grid = {}

    for i in tqdm.tqdm(range(grid_size ** 2)):
        f1s = []
        for file in tqdm.tqdm(files, leave=False):
            _, _, pred_pairings = read_ct(grid_path / str(i) / f"{file.stem}.ct")
            _, _, true_pairings = read_ct(file)

            f1s.append(score_f1(true_pairings, pred_pairings))

        grid[i] = np.nanmean(f1s)

    o = {"grid": grid, "max": max(grid.values()), "argmax": max(grid, key=grid.get)}

    with open(grid_path / "grid.json", "w") as f:
        json.dump(o, f)

    b, m = d[str(o["argmax"])]
    print(
        f"Found optimal F1 at b={b:.1f} kcal/mol, m={m:.1f} kcal/mol. Note: recommend b=-0.6 kcal/mol, m=1.8 kcal/mol."
    )


def _fold_datum(p):
    fps, si, sm = p
    seq_path, save_path, shape_path = fps
    _fold(seq_path, save_path, shape_path, si=si, sm=sm)


def fold(seq_path, shape_path, save_path, si, sm):
    files = list(pathlib.Path(seq_path).glob("*.seq"))

    points = []
    for f in files:
        fps = (
            f,
            save_path / f"{f.stem}.ct",
            shape_path / f"{f.stem}.shape",
        )
        points.append((fps, si, sm))

    with multiprocessing.Pool(config.cpus) as p:
        list(tqdm.tqdm(p.imap_unordered(_fold_datum, points), total=len(points)))
