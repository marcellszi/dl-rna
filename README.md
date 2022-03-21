# Deep learning models for RNA secondary structure prediction
Github repository for the paper:

M. Szikszai, M. Wise, A. Datta, M. Ward, and D.H. Mathews, *‘Deep learning models for RNA secondary structure prediction (probably) do not generalise across families’*, bioRxiv, Mar. 2022


## Abstract
**Motivation:** The secondary structure of RNA is of importance to its function. Over the last few years, several papers attempted to use machine learning to improve *de novo* RNA secondary structure prediction. Many of these papers report impressive results for intra-family predictions, but seldom address the much more difficult (and practical) inter-family problem.

**Results:** We demonstrate it is nearly trivial with convolutional neural networks to generate pseudo-free energy changes, modeled after structure mapping data, that improve the accuracy of structure prediction for intra-family cases. We propose a more rigorous method for inter-family cross-validation that can be used to assess the performance of learning-based models. Using this method, we further demonstrate that intra-family performance is insufficient proof of generalisation despite the widespread assumption in the literature, and provide strong evidence that many existing learning-based models have not generalised inter-family.

## Data
Our model uses [CT (Connectivity Table)](https://rna.urmc.rochester.edu/Text/File_Formats.html#CT) files for secondary structures. For sequences without corresponding secondary structures, the model uses [SEQ](https://rna.urmc.rochester.edu/Text/File_Formats.html#SEQ) files. The predicted SHAPE-like values are stored as [SHAPE Data File Format](https://rna.urmc.rochester.edu/Text/File_Formats.html#SHAPE).

The dataset used by our model is ArchiveII[[2]](#ref2), which can be downloaded [directly from Mathews lab](https://rna.urmc.rochester.edu/pub/archiveII.tar.gz), or from [the release](https://github.com/marcellszi/dl-rna/releases).

Our dataset splits are provided as newline-separated text files containing the filenames (without extension) of the RNAs in each split, made available with [the release](https:/jgithub.com/marcellszi/dl-rna/releases). We also provide tarballs containing CT and SEQ files for our dataset.


## Getting started
### Requirements
Start by downloading RNAstructure[[1]](#ref1). The latest release is available [directly from Mathews lab](https://rna.urmc.rochester.edu/RNAstructure.html).

Next, set up your Python environment. We recommend using [Anaconda](https://www.anaconda.com/distribution/) instead of pip, however, a `requirements.txt` is included.

To install using [Anaconda](https://www.anaconda.com/distribution/):
```
$ conda env create -f environment.yml
```
Next, activate the new environment:
```
$ conda activate dl-rna
```

### Configuration
After installing the requirements, modify `config.json` as needed.

- You will likely need to modify `"rnastructureexe_path"` and `"rnastructuredata_path"` to point to your `RNAstructure/exe` and `RNAstructure/data_tables` locations respectively.
- Ensure your `"device"` is set up correctly. By default, `"device": null` will use the current CUDA device if available, else use CPU. If you wish to change this behaviour, you can pass in an appropriate [torch.device](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) string, such as `cuda:1`.
- Change `"cpus"` to the number of worker processes you want to use for folds (such as during grid-search, or making predictions). By default, `"cpus": null` will use the number returned by [os.cpu_count()](https://docs.python.org/3/library/os.html#os.cpu_count). Please note that the actual folds are *not* GPU accelerated, even if `"device": "cuda"`.


## Training
The training script takes three positional arguments:
- `train_path` - The path to a directory containing CT files used to train the model. Please note that the filenames must end in `.seq`.
- `valid_path` - The path to a directory containing CT files used to validate the model. Please note that the filenames must end in `.seq`. This is always required due to early stopping.
- `output_path` - The path where the model will be output. This will create several sub-directories and save a `model.pt` file.

And two optional arguments:
- `--test_path` - A path to a directory containing CT files used to test the model. This set is also evaluated at the end of each epoch, and saved under `training_statistics.json`.
- `--grid-search` - Grid-search is only performed if this argument is provided. Since we recommend using m=1.8 kcal/mol and b=-0.6 kcal/mol in all cases, grid-search may not be necessary.
```
usage: train.py [-h] [--test_path TEST_PATH] [--grid-search]
                train_path valid_path output_path

train the demonstrative model

positional arguments:
  train_path            path to training CT files
  valid_path            path to validation CT files
  output_path           path where model will be output

optional arguments:
  -h, --help            show this help message and exit
  --test_path TEST_PATH
                        path to testing CT files, optional but will output
                        statistics on test set
  --grid-search         perform grid-search
```

For example, training the model using the family-fold 5S rRNA split provided with the release (including validation and testing sets), and saving to a sub-directory `5s`:
```
$ python train.py data/ct/fam-fold/5s/train data/ct/fam-fold/5s/valid 5s --test_path data/ct/fam-fold/5s/test
```

## Prediction
The prediction script takes three positional arguments:
- `model_path` - The path to the directory containing `model.pt` for your model.
- `seq_path` - The path to the SEQ files you want to predict. Please note that the filenames must end in `.seq`.
- `output_path` - The path where predicted CT and SHAPE files will be output. This will create `ct` and `shape` subdirectories.

And two optional arguments:
- `si` - Intercept used with SHAPE restraints, default: -0.6 kcal/mol.
- `sm` - slope used with SHAPE restraints, default: 1.8 kcal/mol.
```
usage: predict.py [-h] [-si SI] [-sm SM] model_path seq_path output_path

predict using a demonstrative model

positional arguments:
  model_path   path to folder containing `model.pt`
  seq_path     path to testing SEQ files
  output_path  path where CT and SHAPE files will be output

optional arguments:
  -h, --help   show this help message and exit
  -si SI       intercept used with SHAPE restraints, default: -0.6 kcal/mol
  -sm SM       slope used with SHAPE restraints, default: 1.8 kcal/mol
```

For example, testing the model fit in the above example, using the family-fold 5S rRNA test split provided with the release:
```
$ python predict.py 5s data/seq/fam-fold/5s/test 5s
```


## Evaluation
The evaluation script takes three positional arguments:
- `pred_path` - The path to a directory containing the predicted CT files used to evaluate the model. Please note that the filenames must end in `.seq`.
- `true_path` - The path to a directory containing the ground-truth CT files used to evaluate the model. Please note that the filenames must end in `.seq`.
- `output_path` - The path to where the CSV file containing sensitivity, PPV, and F1 values will be written.
```
usage: evaluate.py [-h] pred_path true_path output_path

calculate PPV, sensitivity, and F1 for CT files

positional arguments:
  pred_path    path to predicted CT files
  true_path    path to ground-truth CT files
  output_path  path where CSV will be output

optional arguments:
  -h, --help   show this help message and exit
```

For example, evaluating the data predicted in the above example, and saving to `5s/results.csv`:
```
$ python evaluate.py 5s/ct data/ct/fam-fold/5s/test 5s/results.csv
```

## References
<a name="ref1"></a> - [1] J. S. Reuter and D. H. Mathews, *‘RNAstructure: software for RNA secondary structure prediction and analysis’*, BMC Bioinformatics, vol. 11, no. 1, p. 129, Mar. 2010, doi: [10.1186/1471-2105-11-129](https://doi.org/10.1186/1471-2105-11-129).


<a name="ref2"></a> - [2] M. F. Sloma and D. H. Mathews, *‘Exact calculation of loop formation probability identifies folding motifs in RNA secondary structures’*, RNA, vol. 22, no. 12, pp. 1808–1818, Dec. 2016, doi: [10.1261/rna.053694.115](https://doi.org/10.1261/rna.053694.115).
