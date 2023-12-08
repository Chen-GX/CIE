import os, sys
from utils import *
from train_causal import *


def main():
    args = parse_args()
    data = load_data(args, args.dataset, root=args.data_root, bias_type=args.bias_type, level=args.level)
    model_func = get_model(args)
    train_baseline(model_func, data, args)


if __name__ == '__main__':
    os.chdir(sys.path[0])
    main()
