import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def add_common_args(parser, default_dataset, default_stamp):
    parser.add_argument("--stamp", type=str, default=default_stamp)
    parser.add_argument("--dataset", type=str, default=default_dataset)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--edge_emb_flag", type=str2bool, default=False)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--eva_neg_num", type=int, default=1000)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--tot_epochs", type=int, default=150)
    parser.add_argument("--encoder_epochs", type=int, default=5)
    parser.add_argument("--gsl_epochs", type=int, default=5)
    parser.add_argument("--dropout_adj", type=float, default=1.0)
    parser.add_argument("--lr_encoder", type=float, default=0.005)
    parser.add_argument("--lr_gsl", type=float, default=0.005)
    parser.add_argument("--m_head", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--add_weight", type=float, default=0.5)
    parser.add_argument("--r_add", type=float, default=2)
    parser.add_argument("--r_dele", type=float, default=100.0)
    parser.add_argument("--min_keep", type=float, default=0.1)
    parser.add_argument("--max_add", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cl_lam", type=float, default=0.005)
    parser.add_argument("--pseudo_num", type=int, default=10)
    parser.add_argument("--pseudo_lam", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--valbatch", type=int, default=1024)
    parser.add_argument("--gcl_temp", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--soc_lam", type=float, default=1.0)
    parser.add_argument("--cf_lam", type=float, default=0.8)
    parser.add_argument("--w_o_gate", type=str2bool, default=False)
    parser.add_argument("--w_o_bce", type=str2bool, default=False)
    parser.add_argument("--w_o_hsic", type=str2bool, default=False)
    parser.add_argument("--hsic_lam", type=float, default=0.01)


def yelp_params():
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_dataset="yelp1", default_stamp="ncmfr_yelp")
    parser.add_argument("--param_set", type=str, default="yelp", choices=["yelp", "flickr", "douban"])
    parser.add_argument("--train_iters", type=int, default=3)
    return parser.parse_args()


def flickr_params():
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_dataset="flickr", default_stamp="ncmfr_flickr")
    parser.add_argument("--param_set", type=str, default="flickr", choices=["yelp", "flickr", "douban"])
    parser.set_defaults(
        train_iters=1,
        weight_decay=1e-4,
        dropout_adj=0.5,
        lr_encoder=0.001,
        lr_gsl=0.001,
        r_add=10,
        encoder_epochs=90,
        gsl_epochs=10,
        tot_epochs=100,
        pseudo_num=5,
    )
    parser.add_argument("--train_iters", type=int, default=1)
    return parser.parse_args()


def douban_params():
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_dataset="douban-book", default_stamp="ncmfr_douban")
    parser.add_argument("--param_set", type=str, default="douban", choices=["yelp", "flickr", "douban"])
    parser.add_argument("--train_iters", type=int, default=3)
    return parser.parse_args()


def set_params():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--param_set", type=str, default="yelp", choices=["yelp", "flickr", "douban"])
    known, _ = selector.parse_known_args()
    if known.param_set == "flickr":
        return flickr_params()
    if known.param_set == "douban":
        return douban_params()
    return yelp_params()
