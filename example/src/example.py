import argparse
import torch

from gcs_interpret import task_example


def main_func(args):
    # select device: cpu or gpu
    if args.device < 0:
        device = torch.device("cpu")
    elif args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:  # only for task_xxx_time task
        device = [int(i) for i in args.device.split(",")]
    args.device = device
    task_example(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Convolution Simulator.")
    # GCS model
    parser.add_argument("--device", type=int, default=-1,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate of GCS.")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="number of training epochs.")
    parser.add_argument("--patience", type=int, default=10,
                        help="used for early stop")
    parser.add_argument("--mlp_drop", type=float, default=0.0,
                        help="dropout rate of MLP layers.")
    parser.add_argument("--attn_drop", type=float, default=0.0,
                        help="dropout rate of attention weights.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="temperature of sigmoid function for attention.")
    parser.add_argument("--drop_ratio", type=float, default=0.1,
                        help="attention coeffients dropping threshold")
    parser.add_argument("--data_dir", type=str, default="./example_data",
                        help="dir for load data and save results")
    parser.add_argument("--loss", type=str, default="rc_loss", 
                        help="[rc_loss, mi_loss]")
    parser.add_argument("--visualize", type=bool, default="False", 
                        help="For small KG, you can visualize it in Figure")
    parser.add_argument("--emb_vlm", type=str, default="emb_roberta.pt", 
                        help="Entity label embedding file for vanilla LM")
    parser.add_argument("--emb_klm", type=str, default="emb_kadapter.pt", 
                        help="Entity label embedding file for knowledge-enhanced LM")


    args = parser.parse_args()

    print(args)
    main_func(args)
