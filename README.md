# NCMFR

## Project Structure

```text
NCMFR/
|-- main.py           # Training entry point
|-- model.py          # Main model
|-- encoder.py        # Dual-graph encoder
|-- gsl_uu.py         # Graph structure learning for UU graph
|-- module.py         # Gate and attention modules
|-- layer.py          # MLP and LightGCN propagation layers
|-- dataloader.py     # Dataset loader
|-- evaluation.py     # Evaluation metrics
|-- utils.py          # Loss functions and utilities
|-- parse.py          # Hyperparameter configurations
|-- preprocess.py     # Data preprocessing
|-- data/             # Raw data files
`-- dataset/          # Processed data files
```

## Requirements

Python 3.8+, PyTorch, NumPy, SciPy, PyYAML, pandas

## Training

```bash
# Train on Yelp
python main.py --param_set yelp --dataset yelp1

# Train on Flickr
python main.py --param_set flickr --dataset flickr

# Train on Douban-Book
python main.py --param_set douban --dataset douban-book
```

Each epoch consists of:

1. Encoder training
2. Evaluation on the validation set

The best model and results are saved to `./output/<dataset>/<stamp>/`.
