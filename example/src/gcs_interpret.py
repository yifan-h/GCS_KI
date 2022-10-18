import os
import torch
import dgl
import networkx as nx

from utils import load_data, load_emb, graph_completion, draw_networkx_edge_labels
from gcs_model import gcs_attention


def run_gcs(args, KG, feats_org, feats_klm):
    # cpu for whole KG analysis
    args.device = torch.device("cpu")
    if KG.number_of_nodes() != feats_klm.shape[0]:
        KG = graph_completion(KG)
    KG = dgl.from_networkx(KG)
    KG = dgl.add_self_loop(KG)
    model = gcs_attention(KG, 
                            feats_klm.shape[1], 
                            args.num_heads,
                            args.temperature,
                            args.mlp_drop,
                            args.attn_drop).to(args.device)
    model_path = os.path.join(args.data_dir, "model_gcs.pkl")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    feats_org = torch.FloatTensor(feats_org).to(args.device)
    feats_klm = torch.FloatTensor(feats_klm).to(args.device)
    # start running (training)
    model.train()
    losses = [999. for _ in range(args.patience)]
    for e in range(args.epoch):
        if args.loss == "mi_loss":
            loss, attn = model.mi_loss(feats_org, feats_klm)
        else:
            loss, attn = model.rc_loss(feats_org, feats_klm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e % 10 == 0:
            print("Loss (GCS output, KG-enhanced reps) = ", loss.data.item())
        # early stop
        if loss.data.item() < max(losses):
            losses.remove(max(losses))
            losses.append(loss.data.item())
        else:
            break
    return attn.detach().cpu()


def results_visual(args, G, a, q2l, q2i):
    dgl_g = dgl.from_networkx(G)
    dgl_g = dgl.add_self_loop(dgl_g)
    dgl_g.edata["a"] = a
    nx_g = dgl.to_networkx(dgl_g, edge_attrs=["a"])
    G = nx.DiGraph()
    i2q = {v:k for k, v in q2i.items()}
    for nid in nx_g.nodes():
        G.add_node(nid)
        G.nodes[nid]["label"] = q2l[i2q[str(nid)]]
    for (src, dst) in nx_g.edges():
        G.add_edge(src, dst)
        G[src][dst]["a"] = round(float(nx_g[src][dst][0]["a"]), 3)
    mapping = {i:q2l[i2q[str(i)]] for i in G.nodes()}
    G_l = nx.relabel_nodes(G, mapping)
    G_l.remove_edges_from(nx.selfloop_edges(G_l))  # remove self-loops
    # draw_pos = nx.spring_layout(G_l, scale=20, k=0.9, iterations=50)
    draw_pos = nx.circular_layout(G_l, scale=40)
    # draw_pos = nx.shell_layout(G_l, scale=20)
    G_u = nx.Graph()
    # for e in G_l.edges(data=True): print(e)
    for e in G_l.edges(data=True):
        G_u.add_edge(e[0], e[1])
        if "weight" not in G_u[e[0]][e[1]]:
            G_u[e[0]][e[1]]["weight"] = e[2]["a"]
        else:
            G_u[e[0]][e[1]]["weight"] = (G_u[e[0]][e[1]]["weight"] + e[2]["a"])/2
    # for e in G_u.edges(data=True): print(e)
    # draw the picture
    # draw_g = nx.draw_networkx(G_u, pos=draw_pos, font_size=2, node_size=200, width=0.3, arrows=None)
    draw_g = nx.draw_networkx(G_u, pos=draw_pos, font_size=3, font_weight="bold", node_size=360, width=1e-8)
    for edge in G_u.edges(data='weight'):
        nx.draw_networkx_edges(G_u, pos=draw_pos, edgelist=[edge], width=4*edge[2])
    '''
    # draw the attention weights
    edge_labels = {}
    for src, dst, edata in G_l.edges(data=True):
        if (src, dst) not in edge_labels and (dst, src) not in edge_labels:
            edge_labels[(src, dst)] = edata["a"]
        else:
            if (src, dst) in edge_labels: 
                curr_edata = edge_labels[(src, dst)]
                edge_labels[(src, dst)] = max(edata["a"], curr_edata)
            else: 
                curr_edata = edge_labels[(dst, src)]
                edge_labels[(dst, src)] = max(edata["a"], curr_edata)
    # draw_g = draw_networkx_edge_labels(G_l, pos=draw_pos, font_size=2, edge_labels=edge_labels, label_pos=0.5, arrows=None)
    labels = {(u, v): d for u, v, d in G_l.edges(data=True)}
    '''
    import matplotlib.pyplot as plt
    limits = plt.axis("off")
    plt.savefig(os.path.join(args.data_dir, "results.pdf"), format='pdf', bbox_inches="tight")


def results_save(args, G, a, q2l, q2i):
    dgl_g = dgl.from_networkx(G)
    dgl_g = dgl.add_self_loop(dgl_g)
    dgl_g.edata["a"] = a
    nx_g = dgl.to_networkx(dgl_g, edge_attrs=["a"])
    G = nx.DiGraph()
    i2q = {v:k for k, v in q2i.items()}
    for nid in nx_g.nodes():
        G.add_node(nid)
        G.nodes[nid]["label"] = q2l[i2q[str(nid)]]
    for (src, dst) in nx_g.edges():
        G.add_edge(src, dst)
        G[src][dst]["a"] = float(nx_g[src][dst][0]["a"])
    mapping = {i:i2q[str(i)] for i in G.nodes()}
    G_l = nx.relabel_nodes(G, mapping)
    nx.write_edgelist(G_l, os.path.join(args.data_dir, "gcs.edgelist"))


def task_example(args):
    # load data
    print("start to load data...")
    all_text, all_KG, all_idx = load_data(args)
    all_feats_org, all_feats_klm = load_emb(args)

    attn = run_gcs(args, all_KG, all_feats_org, all_feats_klm)
    if args.visualize:
        results_visual(args, all_KG, attn, all_text, all_idx)
    # save results
    results_save(args, all_KG, attn, all_text, all_idx)

    return

