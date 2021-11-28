import argparse
import torch

from tasks import task_attention_plain, task_attention_drop, task_integration_analysis, \
            task_forgetting_analysis, task_plot, task_robustness, task_baselines


def main_func(args):
    # select device: cpu or gpu
    if args.device < 0:
        device = torch.device("cpu")
    elif args.device >= 0:
        device = torch.device("cuda:" + str(args.device))
    else:  # only for task_xxx_time task
        device = [int(i) for i in args.device.split(",")]
    args.device = device

    # select model to simulate: ernie_thu or kadapter
    if args.simulate_model == "ernie" or "kadapter":
        # task_attention_plain(args)  # obtain attention coefficients for all triples
        # task_attention_drop(args)  # select subset of triples for pretraining
        # task_integration_analysis(args) 
        # task_forgetting_analysis(args)
        task_plot(args)
        # task_robustness(args)
        # task_baselines(args)
    else:
        raise ValueError("Model Name Error!. Assign ernie_thu or kadapter instead!")

    return




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
    parser.add_argument("--mlp_drop", type=float, default=0.2,
                        help="dropout rate of MLP layers.")
    parser.add_argument("--attn_drop", type=float, default=0.0,
                        help="dropout rate of attention weights.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of hidden attention heads.")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="temperature of sigmoid function for attention.")
    parser.add_argument("--drop_ratio", type=float, default=0.1,
                        help="attention coeffients dropping threshold")

    # GCS experiments
    parser.add_argument("--simulate_model", type=str, default="kadapter",
                        help="KG enhanced LM to simulate: [ernie, kadapter].")
    parser.add_argument("--simuate_task", type=str, default="attention",
                        help="task to analyze: [attention, smoothness].")
    parser.add_argument("--simuate_pattern", type=str, default="plain",
                        help="pattern to analyze: [plain, timewise, layerwise].")

    # kadapter
    parser.add_argument("--data_dir", type=str, default="./data/trex-rc",
                        help="The input data dir.")
    parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                        help="The transformer layers of adapter.")
    parser.add_argument("--adapter_size", default=768, type=int,
                        help="The hidden size of adapter.")
    parser.add_argument("--adapter_list", default="0,11,22", type=str,
                        help="The layer where add an adapter.")
    parser.add_argument("--adapter_skip_layers", default=0, type=int,
                        help="The skip_layers of adapter according to bert layers.")
    parser.add_argument("--freeze_bert", default=True, type=bool,
                        help="freeze the parameters of original model.")
    parser.add_argument("--freeze_adapter", default=True, type=bool,
                        help="freeze the parameters of adapter.")
    parser.add_argument("--meta_fac_adaptermodel", default="./pretrained_models/fac_kadapter.bin", type=str, 
                        help="the pretrained factual adapter model.")
    parser.add_argument("--meta_lin_adaptermodel", default="", type=str, 
                        help="the pretrained linguistic adapter model.")
    parser.add_argument("--fusion_mode", type=str, default="concat",
                        help="the fusion mode for bert feature and adapter feature: [add, concat].")

    args = parser.parse_args()
    args.adapter_list = args.adapter_list.split(",")
    args.adapter_list = [int(i) for i in args.adapter_list]

    print(args)

    main_func(args)
