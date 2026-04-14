# NCMFR

## Project Structure

```
NCMFR/
├── main.py           # Training entry point
├── model.py          # Main model (Inac_rec)
├── encoder.py        # Dual-graph encoder (UI + UU propagation)
├── gsl_uu.py         # Graph structure learning for UU graph
├── module.py         # Inter-denoise gate & multi-head attention
├── layer.py          # MLP and LightGCN propagation layers
├── dataloader.py     # Dataset loader (Yelp, Flickr)
├── evaluation.py     # Metrics: NDCG, Hit, Recall, Precision
├── utils.py          # Loss functions and utilities
├── parse.py          # Hyperparameter configurations
└── dataset/          # Data directory (not included)

```

## Requirements

- Python 3.8+, PyTorch, NumPy, SciPy, PyYAML

## Training

```bash
# Train on Yelp (default)
python main.py

# Train on Flickr: change `data` in parse.py to 'flickr', then run
python main.py
```

Each epoch consists of:
1. **Encoder training** — BPR loss on social links + inter-domain denoising loss
2. **Evaluation** — NDCG, Hit, Recall, Precision @10/20 on validation set

Early stopping is based on Recall@10 (patience=20 by default). The best model and results are saved to `./output/<dataset>/<stamp>/`.
