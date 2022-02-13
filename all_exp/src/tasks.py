import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
import networkx as nx
import numpy as np
import heapq
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils import load_data, graph_completion, plot_scatter, plot_distribution, plot_random, plot_baseline,\
    get_edge_idx, deepwalk, dist_node, dist_edge, read_edgelist, binarize2np, learned_edges, plot_biasvar
from emb import load_emb, load_attn, save_attn
from models import gcs_attention, linear_classifier


def run_lc_var(args, G, emb, reps, mlp_layer):
    # load results G
    if mlp_layer:
        if not os.path.exists(os.path.join(args.data_dir, "kg_mlp_var.edgelist")):
            results_G = G.copy()
            for (src, dst) in results_G.edges(): 
                results_G[src][dst]["results"] = {"CF":0, "CR":0, "WL":0}
            nx.write_edgelist(results_G, os.path.join(args.data_dir, "kg_mlp_var.edgelist"))
        results_G = read_edgelist(os.path.join(args.data_dir, "kg_mlp_var.edgelist"))
    else:
        if not os.path.exists(os.path.join(args.data_dir, "kg_lc_var.edgelist")):
            results_G = G.copy()
            for (src, dst) in results_G.edges(): 
                results_G[src][dst]["results"] = {"CF":0, "CR":0, "WL":0}
            nx.write_edgelist(results_G, os.path.join(args.data_dir, "kg_lc_var.edgelist"))
        results_G = read_edgelist(os.path.join(args.data_dir, "kg_lc_var.edgelist"))
    # learning: n*n
    train_nfeat_idx, train_gfeat_idx, train_label = [], [], []
    test_nfeat_idx, test_gfeat_idx, test_label = [], [], []
    for (src, dst) in G.edges():
        if random.random() > 0.5:
            train_nfeat_idx.append(src)
            train_gfeat_idx.append(dst)
            train_label.append(1)
        else:
            test_nfeat_idx.append(src)
            test_gfeat_idx.append(dst)
            test_label.append(1)
    # negative sampling
    for i in range(5*G.number_of_edges()):
        src, dst = random.sample(range(G.number_of_nodes()), 2)
        if (src, dst) in G.edges(): continue
        if random.random() > 0.5:
            train_nfeat_idx.append(src)
            train_gfeat_idx.append(dst)
            train_label.append(0)
        else:
            test_nfeat_idx.append(src)
            test_gfeat_idx.append(dst)
            test_label.append(0)
    train_nfeat_idx = torch.LongTensor(train_nfeat_idx)
    train_gfeat_idx = torch.LongTensor(train_gfeat_idx)
    train_label = torch.FloatTensor(train_label)
    test_nfeat_idx = torch.LongTensor(test_nfeat_idx)
    test_gfeat_idx = torch.LongTensor(test_gfeat_idx)
    test_label = torch.FloatTensor(test_label)
    # emb = torch.FloatTensor(emb)
    # training
    model_org = linear_classifier(emb.shape[1], mlp_layer)
    model_klm = linear_classifier(reps.shape[1], mlp_layer)
    optimizer_org = torch.optim.Adam(model_org.parameters(), lr=1e-4)
    optimizer_klm = torch.optim.Adam(model_klm.parameters(), lr=1e-4)
    loss_fcn_org = torch.nn.MSELoss()
    loss_fcn_klm = torch.nn.MSELoss()
    model_org.train()
    model_klm.train()
    idx_train_sampler=DataLoader([i for i in range(len(train_label))], batch_size=1024, shuffle=True)
    for idx in idx_train_sampler:
        nfeat = emb[train_nfeat_idx[idx]]
        gfeat = emb[train_gfeat_idx[idx]]
        optimizer_org.zero_grad()
        optimizer_klm.zero_grad()
        preds_org = model_org(nfeat, gfeat)
        preds_klm = model_klm(nfeat, gfeat)
        labels = train_label[idx]
        loss_org = loss_fcn_org(preds_org, labels)
        loss_klm = loss_fcn_org(preds_klm, labels)
        loss_org.backward()
        loss_klm.backward()
        optimizer_org.step()
        optimizer_klm.step()
    # test
    model_org.eval()
    model_klm.eval()
    idx_test_sampler=DataLoader([i for i in range(len(test_label))], batch_size=1024, shuffle=True)
    for idx in idx_test_sampler:
        # org
        nfeat = emb[test_nfeat_idx[idx]]
        gfeat = emb[test_gfeat_idx[idx]]
        preds_org = model_org(nfeat, gfeat).detach()
        preds_org_bin = binarize2np(preds_org)
        # klm
        nfeat = reps[test_nfeat_idx[idx]]
        gfeat = reps[test_gfeat_idx[idx]]
        preds_klm = model_klm(nfeat, gfeat).detach()
        labels = test_label[idx].detach()
        preds_klm_bin = binarize2np(preds_klm)
        # get learned number
        labels = binarize2np(labels)
        # assign results in results_G
        results = (preds_klm_bin-preds_org_bin)*labels
        for i in range(labels.shape[0]):
            if labels[i] == 1:
                src = int(test_nfeat_idx[idx[i]])
                dst = int(test_gfeat_idx[idx[i]])
                tmp_results = results_G[src][dst]["results"]
                if results[i] == 0:
                    tmp_results["CR"] = tmp_results["CR"] + 1
                elif results[i] == 1:
                    tmp_results["WL"] = tmp_results["WL"] + 1
                else:
                    tmp_results["CF"] = tmp_results["CF"] + 1
                results_G[src][dst]["results"] = tmp_results
    # reverse training
    model_org = linear_classifier(emb.shape[1], mlp_layer)
    model_klm = linear_classifier(reps.shape[1], mlp_layer)
    optimizer_org = torch.optim.Adam(model_org.parameters(), lr=1e-4)
    optimizer_klm = torch.optim.Adam(model_klm.parameters(), lr=1e-4)
    loss_fcn_org = torch.nn.MSELoss()
    loss_fcn_klm = torch.nn.MSELoss()
    model_org.train()
    model_klm.train()
    idx_test_sampler=DataLoader([i for i in range(len(test_label))], batch_size=1024, shuffle=True)
    for idx in idx_test_sampler:
        nfeat = emb[test_nfeat_idx[idx]]
        gfeat = emb[test_gfeat_idx[idx]]
        optimizer_org.zero_grad()
        optimizer_klm.zero_grad()
        preds_org = model_org(nfeat, gfeat)
        preds_klm = model_klm(nfeat, gfeat)
        labels = test_label[idx]
        loss_org = loss_fcn_org(preds_org, labels)
        loss_klm = loss_fcn_org(preds_klm, labels)
        loss_org.backward()
        loss_klm.backward()
        optimizer_org.step()
        optimizer_klm.step()
    # reverse test
    model_org.eval()
    model_klm.eval()
    idx_train_sampler=DataLoader([i for i in range(len(train_label))], batch_size=1024, shuffle=True)
    for idx in idx_train_sampler:
        # org
        nfeat = emb[train_nfeat_idx[idx]]
        gfeat = emb[train_gfeat_idx[idx]]
        preds_org = model_org(nfeat, gfeat).detach()
        preds_org_bin = binarize2np(preds_org)
        # klm
        nfeat = reps[train_nfeat_idx[idx]]
        gfeat = reps[train_gfeat_idx[idx]]
        preds_klm = model_klm(nfeat, gfeat).detach()
        labels = train_label[idx].detach()
        preds_klm_bin = binarize2np(preds_klm)
        # get learned number
        labels = binarize2np(labels)
        # assign results in results_G
        results = (preds_klm_bin-preds_org_bin)*labels
        for i in range(labels.shape[0]):
            if labels[i] == 1:
                src = int(train_nfeat_idx[idx[i]])
                dst = int(train_gfeat_idx[idx[i]])
                tmp_results = results_G[src][dst]["results"]
                if results[i] == 0:
                    tmp_results["CR"] = tmp_results["CR"] + 1
                elif results[i] == 1:
                    tmp_results["WL"] = tmp_results["WL"] + 1
                else:
                    tmp_results["CF"] = tmp_results["CF"] + 1
                results_G[src][dst]["results"] = tmp_results
    # save results
    if mlp_layer:
        nx.write_edgelist(results_G, os.path.join(args.data_dir, "kg_mlp_var.edgelist"))
    else:
        nx.write_edgelist(results_G, os.path.join(args.data_dir, "kg_lc_var.edgelist"))
    return


def task_lc_var(args):
    if not os.path.exists(os.path.join(args.data_dir, "kg_lc_var.edgelist")) or\
        not os.path.exists(os.path.join(args.data_dir, "kg_mlp_var.edgelist")):
        # load embedding
        all_KG_nx = read_edgelist(os.path.join(args.data_dir, "all_KG.edgelist"))
        all_KG_nx = graph_completion(all_KG_nx)
        if args.simulate_model == "kadapter": org_model_name = "roberta-large"
        else: org_model_name = "bert-base-uncased"
        all_feats_org = load_emb(args, None, None, org_model_name, "_all")
        all_feats_klm = load_emb(args, None, None, args.simulate_model, "_all")
        all_KG = dgl.from_networkx(all_KG_nx)
        all_KG = dgl.add_self_loop(all_KG)
        src, dst, eid = all_KG.edges(form="all")

        # linear probe
        for i in tqdm(range(100)):
            run_lc_var(args, all_KG_nx, all_feats_org, all_feats_klm, mlp_layer=False)
        
        # MLP probe
        for i in tqdm(range(100)):
            run_lc_var(args, all_KG_nx, all_feats_org, all_feats_klm, mlp_layer=True)

    results_lc = read_edgelist(os.path.join(args.data_dir, "kg_lc_var.edgelist"))
    results_mlp = read_edgelist(os.path.join(args.data_dir, "kg_mlp_var.edgelist"))
    plot_biasvar(args, results_lc, os.path.join(args.data_dir, "results/lc_var.pdf"))
    plot_biasvar(args, results_mlp, os.path.join(args.data_dir, "results/mlp_var.pdf"), "1-layer MLP")

    return


def run_lc(args, G, emb, reps=None, edges_cos=None, edges_euc=None):
    '''
def run_lc(args, G, noisy_emb, kg_emb):
    # forgetting: positive: nid, negative: eid
    train_nfeat_idx, train_gfeat_idx, train_label = [], [], []
    test_nfeat_idx, test_gfeat_idx, test_label = [], [], []
    for nid in G.nodes():
        if random.random() > 0.5:
            train_nfeat_idx.append(nid)
            train_gfeat_idx.append(nid)
            train_label.append(1)
        else:
            test_nfeat_idx.append(nid)
            test_gfeat_idx.append(nid)
            test_label.append(1)
    for (src, dst) in G.edges():
        if random.random() > 0.5:
            train_nfeat_idx.append(src)
            train_gfeat_idx.append(dst)
            train_label.append(0)
        else:
            test_nfeat_idx.append(src)
            test_gfeat_idx.append(dst)
            test_label.append(0)
    train_nfeat_idx = torch.LongTensor(train_nfeat_idx)
    train_gfeat_idx = torch.LongTensor(train_gfeat_idx)
    train_label = torch.FloatTensor(train_label)
    test_nfeat_idx = torch.LongTensor(test_nfeat_idx)
    test_gfeat_idx = torch.LongTensor(test_gfeat_idx)
    test_label = torch.FloatTensor(test_label)
    noisy_emb = torch.FloatTensor(noisy_emb)
    kg_emb = torch.FloatTensor(kg_emb)
    # training
    model = linear_classifier(noisy_emb.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = torch.nn.MSELoss()
    idx_sampler=DataLoader([i for i in range(len(train_label))], batch_size=1024, shuffle=True)
    model.train()
    for idx in tqdm(idx_sampler):
        # nfeat_idx = train_nfeat_idx[idx]
        # gfeat_idx = train_gfeat_idx[idx]
        # print(type(idx), type(train_nfeat_idx), type(noisy_emb))
        # print(idx.shape, train_nfeat_idx.shape, noisy_emb.shape)
        nfeat = noisy_emb[train_nfeat_idx[idx]]
        gfeat = kg_emb[train_gfeat_idx[idx]]
        optimizer.zero_grad()
        preds = model(nfeat, gfeat)
        labels = train_label[idx]
        loss = loss_fcn(preds, labels)
        loss.backward()
        optimizer.step()
    # test
    idx_sampler=DataLoader([i for i in range(len(test_label))], batch_size=1024, shuffle=True)
    accs, aucs, f1s = [], [], []
    model.eval()
    for idx in tqdm(idx_sampler):
        # nfeat_idx = test_nfeat_idx[idx]
        # gfeat_idx = test_gfeat_idx[idx]
        nfeat = noisy_emb[test_nfeat_idx[idx]]
        gfeat = kg_emb[test_gfeat_idx[idx]]
        preds = model(nfeat, gfeat).detach()
        labels = test_label[idx].detach()
        preds = binarize2np(preds)
        labels = binarize2np(labels)
        accs.append(accuracy_score(labels, preds))
        aucs.append(roc_auc_score(labels, preds))
        f1s.append(f1_score(labels, preds))
    print("Forgetting: linear classifier: (acc, auc, f1)", sum(accs)/len(accs), sum(aucs)/len(aucs), sum(f1s)/len(f1s))
    '''
    # learning: n*n
    train_nfeat_idx, train_gfeat_idx, train_label = [], [], []
    test_nfeat_idx, test_gfeat_idx, test_label = [], [], []
    for (src, dst) in G.edges():
        if random.random() > 0.5:
            train_nfeat_idx.append(src)
            train_gfeat_idx.append(dst)
            train_label.append(1)
        else:
            test_nfeat_idx.append(src)
            test_gfeat_idx.append(dst)
            test_label.append(1)
    # negative sampling
    for i in range(5*G.number_of_edges()):
        src, dst = random.sample(range(G.number_of_nodes()), 2)
        if (src, dst) in G.edges(): continue
        if random.random() > 0.5:
            train_nfeat_idx.append(src)
            train_gfeat_idx.append(dst)
            train_label.append(0)
        else:
            test_nfeat_idx.append(src)
            test_gfeat_idx.append(dst)
            test_label.append(0)
    train_nfeat_idx = torch.LongTensor(train_nfeat_idx)
    train_gfeat_idx = torch.LongTensor(train_gfeat_idx)
    train_label = torch.FloatTensor(train_label)
    test_nfeat_idx = torch.LongTensor(test_nfeat_idx)
    test_gfeat_idx = torch.LongTensor(test_gfeat_idx)
    test_label = torch.FloatTensor(test_label)
    emb = torch.FloatTensor(emb)
    # training
    model_org = linear_classifier(emb.shape[1])
    optimizer = torch.optim.Adam(model_org.parameters(), lr=1e-4)
    loss_fcn = torch.nn.MSELoss()
    model_org.train()
    idx_train_sampler=DataLoader([i for i in range(len(train_label))], batch_size=1024, shuffle=True)
    for idx in tqdm(idx_train_sampler):
        nfeat = emb[train_nfeat_idx[idx]]
        gfeat = emb[train_gfeat_idx[idx]]
        optimizer.zero_grad()
        preds = model_org(nfeat, gfeat)
        labels = train_label[idx]
        loss = loss_fcn(preds, labels)
        loss.backward()
        optimizer.step()
    if reps is not None: 
        reps = torch.FloatTensor(reps)
        # training
        model_klm = linear_classifier(reps.shape[1])
        optimizer = torch.optim.Adam(model_klm.parameters(), lr=1e-4)
        loss_fcn = torch.nn.MSELoss()
        model_klm.train()
        for idx in tqdm(idx_train_sampler):
            nfeat = reps[train_nfeat_idx[idx]]
            gfeat = reps[train_gfeat_idx[idx]]
            optimizer.zero_grad()
            preds = model_klm(nfeat, gfeat)
            labels = train_label[idx]
            loss = loss_fcn(preds, labels)
            loss.backward()
            optimizer.step()
    # test
    aucs_org, aucs_klm, f1s_org, f1s_klm = [], [], [], []
    model_org.eval()
    model_klm.eval()
    idx_test_sampler=DataLoader([i for i in range(len(test_label))], batch_size=1024, shuffle=True)
    total_count, cos_count, euc_count = 0, 0, 0
    for idx in tqdm(idx_test_sampler):
        # org
        nfeat = emb[test_nfeat_idx[idx]]
        gfeat = emb[test_gfeat_idx[idx]]
        preds_org = model_org(nfeat, gfeat).detach()
        preds_org_bin = binarize2np(preds_org)
        # klm
        nfeat = reps[test_nfeat_idx[idx]]
        gfeat = reps[test_gfeat_idx[idx]]
        preds_klm = model_klm(nfeat, gfeat).detach()
        labels = test_label[idx].detach()
        preds_klm_bin = binarize2np(preds_klm)
        # get learned number
        labels = binarize2np(labels)
        if edges_cos is not None:
            results = (preds_klm_bin-preds_org_bin)*labels
            for i in range(labels.shape[0]):
                if results[i] == 1:
                    total_count +=1
                    src = int(test_nfeat_idx[idx[i]])
                    dst = int(test_gfeat_idx[idx[i]])
                    if (src, dst) in edges_cos or (dst, src) in edges_cos:
                        cos_count += 1
                    if (src, dst) in edges_euc or (dst, src) in edges_euc:
                        euc_count += 1
        aucs_org.append(roc_auc_score(labels, preds_org))
        aucs_klm.append(roc_auc_score(labels, preds_klm))
        f1s_org.append(f1_score(labels, preds_org_bin))
        f1s_klm.append(f1_score(labels, preds_klm_bin))
    print("linear classifier performance org: (auc, f1)", sum(aucs_org)/len(aucs_org), sum(aucs_klm)/len(aucs_klm), 
            sum(f1s_org)/len(f1s_org), sum(f1s_klm)/len(f1s_klm),\
            sum(aucs_klm)/len(aucs_klm)-sum(aucs_org)/len(aucs_org), sum(f1s_klm)/len(f1s_klm)-sum(f1s_org)/len(f1s_org))
    print(total_count, cos_count, euc_count)
    return


def run_gcs(args, KG, feats_org, feats_klm, mode="test", save_a=True, nr=0):
    # cpu for whole KG analysis
    tmp_device = args.device
    if mode == "analysis": args.device = torch.device("cpu")
    if KG.number_of_nodes() != feats_klm.shape[0]:
        KG = graph_completion(KG)
    KG = dgl.from_networkx(KG)
    # set Graph Convolution Simulator
    # KG = dgl.from_networkx(KG).to(args.device)
    # self-loop
    if save_a:
        KG = dgl.add_self_loop(KG)
    model = gcs_attention(KG, 
                            feats_klm.shape[1], 
                            args.num_heads,
                            args.temperature,
                            args.mlp_drop,
                            args.attn_drop).to(args.device)
    model_path = os.path.join(args.data_dir, "results", "model_gcs.pkl")
    if mode != "train":
        model.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if feats_org is not None:
        feats_org = torch.FloatTensor(feats_org).to(args.device)
    feats_klm = torch.FloatTensor(feats_klm).to(args.device)
    # set mode
    if mode == "train": 
        model.train()
        epoch_num = args.epoch
    else: 
        model.eval()
        epoch_num = 1

    # start running (training / test)
    mi_es = [-99 for _ in range(args.patience)]
    for e in range(epoch_num):
        if feats_org is None: feats_org = nr*torch.rand(feats_klm.shape)+(1-nr)*feats_klm
        loss, attn = model.mi_loss(feats_org, feats_klm)
        # loss, attn = model.rc_loss(feats_org, feats_klm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("MI(GCS output, KG-enhanced reps) = ", -loss.data.item())
        # early stop
        if -loss.data.item() > min(mi_es):
            mi_es.remove(min(mi_es))
            mi_es.append(-loss.data.item())
        else:
            break

    # save results
    if mode == "train":
        torch.save(model.state_dict(), model_path)
    if save_a:
        save_attn(args, attn, mode)
    args.device = tmp_device

    return attn.detach().cpu()


def task_attention_plain(args):
    # load data
    print("start to load data...")
    all_text, all_KG, all_idx, _ = load_data(args, all_data=True)
    if args.simulate_model == "kadapter": 
        org_model_name = "roberta-large"
        all_feats_org = load_emb(args, all_text, all_idx, org_model_name, "_all")
        all_feats_klm = load_emb(args, all_text, all_idx, args.simulate_model, "_all")
        run_gcs(args, all_KG, all_feats_org, all_feats_klm, mode="all")
    else: 
        org_model_name = "bert-base-uncased"
        train_text, train_KG, train_idx, test_text, test_KG, test_idx = load_data(args)
        train_feats_klm = load_emb(args, train_text, train_idx, args.simulate_model, "_train")
        test_feats_klm = load_emb(args, test_text, test_idx, args.simulate_model, "_test")
        train_feats_org = load_emb(args, train_text, train_idx, org_model_name, "_train")
        test_feats_org = load_emb(args, test_text, test_idx, org_model_name, "_test")
        all_feats_org = load_emb(args, all_text, all_idx, org_model_name, "_all")
        all_feats_klm = load_emb(args, all_text, all_idx, args.simulate_model, "_all")
        # training
        run_gcs(args, train_KG, train_feats_org, train_feats_klm, mode="train")
        # test & analysis
        run_gcs(args, test_KG, test_feats_org, test_feats_klm, mode="test")
        run_gcs(args, all_KG, all_feats_org, all_feats_klm, mode="analysis")

    return


def task_attention_drop(args):
    # load KG, index, attention_coefficients
    _, all_KG, all_idx, all_data = load_data(args, all_data=True)
    """
    # test consistence (networkx graph): pass
    bad_count = 0
    all_set = set()
    for src, dst in all_KG.edges():
        all_set.add(str(src) + "_" + str(dst))
        all_set.add(str(dst) + "_" + str(src))
        all_set.add(str(dst) + "_" + str(dst))
        all_set.add(str(src) + "_" + str(src))
    for d in all_data:
        src_id = all_idx.get(d.get("subj_label"))
        dst_id = all_idx.get(d.get("obj_label"))
        if str(src_id) + "_" + str(dst_id) not in all_set:
            bad_count += 1
    if bad_count: print("warning: bad count number (nx): ", bad_count)
    """
    all_attn = load_attn(args, mode="analysis")
    all_KG = graph_completion(all_KG)
    all_KG = dgl.from_networkx(all_KG)
    all_KG = dgl.add_self_loop(all_KG)
    all_KG.edata["a"] = all_attn
    src, dst, eid = all_KG.edges(form="all")
    """
    # test consistence (dgl graph): pass
    bad_count = 0
    all_set = set()
    for i in range(eid.shape[0]):
        all_set.add(str(int(src[i])) + "_" + str(int(dst[i])))
        all_set.add(str(int(dst[i])) + "_" + str(int(src[i])))
        all_set.add(str(int(dst[i])) + "_" + str(int(dst[i])))
        all_set.add(str(int(src[i])) + "_" + str(int(src[i])))
    for d in all_data:
        src_id = all_idx.get(d.get("subj_label"))
        dst_id = all_idx.get(d.get("obj_label"))
        if str(src_id) + "_" + str(dst_id) not in all_set:
            bad_count += 1
    if bad_count: print("warning: bad count number (dgl): ", bad_count)
    """

    # remove triples with small attention coefficients
    if args.simulate_model == "kadapter":
        remove_set = set()
        for i in range(eid.shape[0]):
            if all_KG.edata["a"][eid[i]] <= args.drop_ratio:
                remove_set.add(str(int(src[i])) + "_" + str(int(dst[i])))
        print(len(remove_set))
        sub_data = []  # useful data
        for d in all_data:
            src_id = all_idx[d["subj_label"]]
            dst_id = all_idx[d["obj_label"]]
            if str(src_id) + "_" + str(dst_id) not in remove_set:
            # and (dst_id, src_id) not in remove_set\
            # and src_id != dst_id:
                sub_data.append(d)
        # save sub_data
        with open(os.path.join(args.data_dir, str(args.drop_ratio) + "_all_data_remove.json"), "w") as f: 
            f.write(json.dumps(sub_data))
        print("start to save cleaned subset of training triples: # org; # cleaned: ", len(all_data), len(sub_data))
    else:
        remove_gid = set()
        for i in range(eid.shape[0]):
            if src[i] == dst[i]:
                if all_KG.edata["a"][eid[i]] > 1 - args.drop_ratio:
                    remove_gid.add(int(dst[i]))
        with open(os.path.join(args.data_dir, "all_idx.json"), "r") as f: 
            all_idx = json.loads(f.read())  # Q-label: KG index
        remove_qid = set()
        print(len(remove_gid))
        for k, v in all_idx.items():
            if v in remove_gid:
                remove_qid.add(k)
        remove_vid = set()
        entity2id, id2entity= {}, {}
        with open(os.path.join(args.data_dir, "entity2id.txt")) as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                entity2id[qid] = int(eid)
                id2entity[int(eid)] = qid
        rest_vid = []
        for qid, eid in entity2id.items():
            if qid in remove_qid:
                remove_vid.add(eid)
            else:
                if qid in all_idx:
                    rest_vid.append(eid)
        # get entity2id.txt
        new_entity2id = {}
        for i in range(len(rest_vid)):
            new_entity2id[id2entity[rest_vid[i]]] = i
        # write entity2id.txt
        with open(os.path.join(args.data_dir, "entity2id.txt"), "w") as fin:
            fin.write(str(len(new_entity2id)))
            for k, v in new_entity2id.items():
                fin.write("\n"+k+"\t"+str(v))
        # get vecs
        vecs = []
        vecs.append([0]*100)
        with open(os.path.join(args.data_dir, "entity2vec.vec"), 'r') as fin:
            for line in fin:
                vec = line.strip().split('\t')
                vecs.append(vec)
        rest_vecs = []
        for i in range(len(rest_vid)):
            rest_vecs.append(vecs[rest_vid[i]])
        # write vecs
        with open(os.path.join(args.data_dir, "entity2vec.vec"), 'w') as fin:
            for vec in rest_vecs:
                vec = "\t".join(str(v) for v in vec)
                fin.write(vec)
                fin.write("\n")
        print("start to save cleaned subset of training triples: # org; # cleaned: ", len(entity2id), len(rest_vid))


    """
    # corupt useless triples
    corrupt_count = 0
    sub_data = []
    for d in all_data:
        src_id = all_idx.get(d.get("subj_label"))
        dst_id = all_idx.get(d.get("obj_label"))
        if str(src_id) + "_" + str(dst_id) in remove_set:
            ids = list(range(len(d.get("token"))))
            random.shuffle(ids)
            d["subj_start"] = ids[0]
            d["subj_end"] = ids[0]
            d["obj_start"] = ids[1]
            d["obj_end"] = ids[1]
            corrupt_count += 1
        sub_data.append(d)

    # save sub_data
    with open(os.path.join(args.data_dir, str(args.drop_ratio) + "_all_data_corupt.json"), "w") as f: 
        f.write(json.dumps(sub_data))

    print("start to save cleaned subset of training triples: # org; # cleaned: ", len(all_data), corrupt_count)
    """

    return


def task_integration_analysis(args):
    # load KG, index, attention_coefficients
    _, all_KG_nx, all_idx, all_data = load_data(args, all_data=True)
    all_attn = load_attn(args, mode="analysis")
    all_KG_nx = graph_completion(all_KG_nx)
    all_KG = dgl.from_networkx(all_KG_nx)
    all_KG = dgl.add_self_loop(all_KG)
    all_KG.edata["a"] = all_attn
    src, dst, eid = all_KG.edges(form="all")
    src_list = src.tolist()
    dst_list = dst.tolist()
    src_idx_dict = {}
    dst_idx_dict = {}
    for i in range(len(src_list)):
        if src_list[i] not in src_idx_dict:
            src_idx_dict[src_list[i]] = set()
        src_idx_dict[src_list[i]].add(i)
        if dst_list[i] not in dst_idx_dict:
            dst_idx_dict[dst_list[i]] = set()
        dst_idx_dict[dst_list[i]].add(i)

    # get attn & distribution
    self_attn_list = []
    for i in range(len(src_list)):
        if src_list[i] == dst_list[i]:
            tmp_a = float(all_KG.edata["a"][eid[i]])
            self_attn_list.append(tmp_a)
    print("start to analyze the distribution of self-loop attns: ", np.histogram(self_attn_list, bins=10), sum(self_attn_list)/len(self_attn_list))

    # analyze attention coefficients
    if args.simulate_model == "kadapter":
        """
        # triple statistics (P-label)
        relation_label = {}
        for d in all_data:
            tmp_r = d.get("relation")
            if tmp_r in relation_label:
                tmp_count = relation_label[tmp_r]
                relation_label[tmp_r] = tmp_count + 1
            else:
                relation_label[tmp_r] = 1
        print("Triple statistics: ", relation_label)
        """

        # relation: data side
        print("start to analyze relation: data side (K-Adapter)...")
        label_count = {"P19": 0, "P361":0, "P570":0, "P569":0, "P131":0, "P17":0}
        label_attn = {}
        remove_set = set()
        for i in range(eid.shape[0]):
            if all_KG.edata["a"][eid[i]] <= args.drop_ratio:
                remove_set.add(str(int(src[i])) + "_" + str(int(dst[i])))
        remove_data = []  # data to drop
        for d in all_data:
            src_id = all_idx[d["subj_label"]]
            dst_id = all_idx[d["obj_label"]]
            if str(src_id) + "_" + str(dst_id) in remove_set:
                remove_data.append(d)
        for d in remove_data:
            if d["relation"] in label_count:
                label_count[d["relation"]] = label_count[d["relation"]] + 1

        # calculate avg attention weights
        all_attn_data = []
        for d in all_data:
            src_id = all_idx[d["subj_label"]]
            dst_id = all_idx[d["obj_label"]]
            tmp_eidx = dst_idx_dict[dst_id].intersection(src_idx_dict[src_id])
            if len(tmp_eidx) != 1: 
                print("warning! dgl multiple edges: ", tmp_eidx)
            else:
                for e in tmp_eidx: tmp_eidx = e
            tmp_a = float(all_KG.edata["a"][eid[tmp_eidx]])
            all_attn_data.append(tmp_a)

        for k, _ in label_count.items():
            tmp_list = []
            for d in all_data:
                if d["relation"] != k:
                    continue
                src_id = all_idx[d["subj_label"]]
                dst_id = all_idx[d["obj_label"]]
                tmp_eidx = dst_idx_dict[dst_id].intersection(src_idx_dict[src_id])
                if len(tmp_eidx) != 1: 
                    print("warning! dgl multiple edges: ", tmp_eidx)
                else:
                    for e in tmp_eidx: tmp_eidx = e
                tmp_a = float(all_KG.edata["a"][eid[tmp_eidx]])
                tmp_list.append(tmp_a)
            label_attn[k] = sum(tmp_list) / len(tmp_list)
        print("....label count and average attention value: ", label_count, label_attn)
        print("....all triple num, dropped triple num, and average attention value: ",  
                                                                            len(all_data),
                                                                            len(remove_data),
                                                                            sum(all_attn_data)/len(all_attn_data))

        # relation: KG side
        print("start to analyze relation: KG side (K-Adapter)...")
        # check id consistence
        for i in range(len(src_list)):
            src_id = src_list[i]
            dst_id = dst_list[i]
            if (src_id, dst_id) in all_KG_nx.edges():
                continue
            elif src_id == dst_id:
                continue
            else:
                raise ValueError("Error! inconsistent DGL graph and networkx graph: ", src_id, dst_id)
        # get 1-1, N-1, N-M relations
        one2one, one2n, n2n = set(), set(), set()
        for nid in all_KG_nx.nodes():
            nid_neighbors = [n for n in all_KG_nx.neighbors(nid)]
            if len(nid_neighbors) == 1:
                nid_neighbor = nid_neighbors[0]
                nid_neighbors_2hops = [n for n in all_KG_nx.neighbors(nid_neighbor)]
                if len(nid_neighbors_2hops) == 1:
                    one2one.add(nid)
                else:
                    one2n.add(nid)
            else:
                n2n.add(nid)
        rel_11_idx, rel_n1_idx, rel_nn_idx = set(), set(), set()
        rel_11_attn, rel_n1_attn, rel_nn_attn = [], [], []
        left_edge_count = {"1-1":0, "N-1":0, "N-M":0}
        print("....the number of 1-1, N-1, N-M nodes are:", len(one2one), len(one2n), len(n2n))
        for i in range(len(src_list)):
            # remove self-loops
            if src_list[i] == dst_list[i]: continue
            if src_list[i] in one2one or dst_list[i] in one2one:
                rel_11_idx.add(i)
                tmp_a = float(all_KG.edata["a"][eid[i]])
                rel_11_attn.append(tmp_a)
                if tmp_a > args.drop_ratio:
                    left_edge_count["1-1"] += 1
            elif src_list[i] in one2n or dst_list[i] in one2n:
                rel_n1_idx.add(i)
                tmp_a = float(all_KG.edata["a"][eid[i]])
                rel_n1_attn.append(tmp_a)
                if tmp_a > args.drop_ratio:
                    left_edge_count["N-1"] += 1
            else:
                rel_nn_idx.add(i)
                tmp_a = float(all_KG.edata["a"][eid[i]])
                rel_nn_attn.append(tmp_a)
                if tmp_a > args.drop_ratio:
                    left_edge_count["N-M"] += 1
        print("....the number of 1-1, N-1, N-M relations are:", len(rel_11_idx), 
                                                                len(rel_n1_idx), 
                                                                len(rel_nn_idx))
        print("....the number of left 1-1, N-1, N-M relations are:", left_edge_count)
        print("....the average attention weights of 1-1, N-1, N-M relations are:", sum(rel_11_attn)/len(rel_11_attn), 
                                                                sum(rel_n1_attn)/len(rel_n1_attn), 
                                                                sum(rel_nn_attn)/len(rel_nn_attn))

        # relation: aligned sentence number
        print("start to analyze relation: training triple frequency side (K-Adapter)...")
        edge_count = [0 for i in range(len(src_list))]
        for d in all_data:
            src_id = all_idx[d["subj_label"]]
            dst_id = all_idx[d["obj_label"]]
            tmp_eidx = dst_idx_dict[dst_id].intersection(src_idx_dict[src_id])
            if len(tmp_eidx) != 1: 
                print("warning! dgl multiple edges: ", tmp_eidx)
            else:
                for e in tmp_eidx: tmp_eidx = e
            edge_count[tmp_eidx] += 1
        all_attn = all_attn.tolist()
        # remove error value
        tmp_edge_count, tmp_all_attn = [], []
        for i in range(len(edge_count)):
            if edge_count[i] == 0 or src_list[i] == dst_list[i] or edge_count[i] > 1000:
                continue
            else:
                tmp_edge_count.append(edge_count[i])
                tmp_all_attn.append(all_attn[i])
        edge_count = np.array(tmp_edge_count)
        all_attn = np.array(tmp_all_attn)
        from scipy import stats
        print(np.corrcoef(all_attn, edge_count), np.cov(all_attn, edge_count), stats.spearmanr(all_attn, edge_count))
        corr_value = np.corrcoef(all_attn, edge_count)[0][1]
        plot_scatter(os.path.join(args.data_dir, "results", "edge_count.pdf"), all_attn, edge_count, corr_value)
    else:  # ERNIE
        # get 1-1, N-1, N-M relations
        print("start to analyze relation: KG side (ERNIE)...")
        # check id consistence
        for i in range(len(src_list)):
            src_id = src_list[i]
            dst_id = dst_list[i]
            if (src_id, dst_id) in all_KG_nx.edges():
                continue
            elif src_id == dst_id:
                continue
            else:
                raise ValueError("Error! inconsistent DGL graph and networkx graph: ", src_id, dst_id)
        # get all nodes
        one2one, one2n, n2n = set(), set(), set()
        for nid in tqdm(all_KG_nx.nodes()):
            nid_neighbors = [n for n in all_KG_nx.neighbors(nid)]
            if len(nid_neighbors) == 1:
                nid_neighbor = nid_neighbors[0]
                nid_neighbors_2hops = [n for n in all_KG_nx.neighbors(nid_neighbor)]
                if len(nid_neighbors_2hops) == 1:
                    one2one.add(nid)
                else:
                    one2n.add(nid)
            elif len(nid_neighbors) < 1:
                continue
            else:
                n2n.add(nid)
        print("....the number of 1-1, N-1, N-M nodes are:", len(one2one), len(one2n), len(n2n))
        left_node_count = {"1-1":0, "N-1":0, "N-M":0}
        rel_11_attn, rel_n1_attn, rel_nn_attn = [], [], []
        # get remove number
        isolated_count = 0
        for i in range(len(src_list)):
            if src_list[i] == dst_list[i]:
                tmp_a = all_KG.edata["a"][eid[i]]
                if tmp_a > 1 - args.drop_ratio:
                    continue
                else:
                    if src_list[i] in one2one:
                        left_node_count["1-1"] += 1
                        rel_11_attn.append(float(tmp_a))
                    elif src_list[i] in one2n:
                        left_node_count["N-1"] += 1
                        rel_n1_attn.append(float(tmp_a))
                    elif src_list[i] in n2n:
                        left_node_count["N-M"] += 1
                        rel_nn_attn.append(float(tmp_a))
                    else:
                        isolated_count += 1
        print("....the number of isolated nodes are:", isolated_count)
        print("....the number of left 1-1, N-1, N-M nodes are:", left_node_count)
        print("....the average attention weights of 1-1, N-1, N-M relations are:", sum(rel_11_attn)/len(rel_11_attn), 
                                                                sum(rel_n1_attn)/len(rel_n1_attn), 
                                                                sum(rel_nn_attn)/len(rel_nn_attn))

    return


def task_forgetting_analysis(args):
    # load KG, index, attention_coefficients
    _, all_KG_nx, all_idx, all_data = load_data(args, all_data=True)
    all_attn = load_attn(args, mode="analysis")
    all_KG_nx = graph_completion(all_KG_nx)
    all_KG = dgl.from_networkx(all_KG_nx)
    all_KG = dgl.add_self_loop(all_KG)
    all_KG.edata["a"] = all_attn
    src, dst, eid = all_KG.edges(form="all")
    src = src.tolist()
    dst = dst.tolist()
    src_idx_dict = {}
    dst_idx_dict = {}

    # get attn & distribution
    self_loop_idx, unlearned_idx, forgetting_idx, welllearned_idx = [], [], [], []
    for i in range(len(src)):
        if src[i] == dst[i]:
            self_loop_idx.append(i)
            if all_KG.edata["a"][i] > 0.9:
                unlearned_idx.append(i)
            elif all_KG.edata["a"][i] < 0.1:
                forgetting_idx.append(i)
            elif all_KG.edata["a"][i] < 0.6 and all_KG.edata["a"][i] > 0.4:
                welllearned_idx.append(i)
    print("start to analyze forgetting: ", len(self_loop_idx), len(unlearned_idx), len(forgetting_idx))

    # get 1-1, N-1, N-M relations (unlearned & forgetting)
    print("start to analyze relation: KG side: ", args.simulate_model)
    # check id consistence
    for i in range(len(src)):
        src_id = src[i]
        dst_id = dst[i]
        if (src_id, dst_id) in all_KG_nx.edges():
            continue
        elif src_id == dst_id:
            continue
        else:
            raise ValueError("Error! inconsistent DGL graph and networkx graph: ", src_id, dst_id)
    # get all nodes
    one2one, one2n, n2n = set(), set(), set()
    for nid in tqdm(all_KG_nx.nodes()):
        nid_neighbors = [n for n in all_KG_nx.neighbors(nid)]
        if len(nid_neighbors) == 1:
            nid_neighbor = nid_neighbors[0]
            nid_neighbors_2hops = [n for n in all_KG_nx.neighbors(nid_neighbor)]
            if len(nid_neighbors_2hops) == 1:
                one2one.add(nid)
            else:
                one2n.add(nid)
        elif len(nid_neighbors) < 1:
            continue
        else:
            n2n.add(nid)
    print("....the number of 1-1, N-1, N-M nodes are:", len(one2one), len(one2n), len(n2n))
    unlearned_node_count = {"1-1":0, "N-1":0, "N-M":0}
    forgetting_node_count = {"1-1":0, "N-1":0, "N-M":0}
    # get remove number
    isolated_count = 0
    for i in self_loop_idx:
        tmp_a = all_KG.edata["a"][eid[i]]
        if tmp_a > 1 - args.drop_ratio:  # unlearned
            if src[i] in one2one:
                unlearned_node_count["1-1"] += 1
            elif src[i] in one2n:
                unlearned_node_count["N-1"] += 1
            elif src[i] in n2n:
                unlearned_node_count["N-M"] += 1
            else:
                isolated_count += 1
        elif tmp_a < args.drop_ratio:  # forgetting
            if src[i] in one2one:
                forgetting_node_count["1-1"] += 1
            elif src[i] in one2n:
                forgetting_node_count["N-1"] += 1
            elif src[i] in n2n:
                forgetting_node_count["N-M"] += 1
            else:
                isolated_count += 1
    print("....the number of isolated nodes are:", isolated_count)
    print("....the number of unlearned 1-1, N-1, N-M nodes are:", unlearned_node_count)
    print("....the number of forgetting 1-1, N-1, N-M nodes are:", forgetting_node_count)

    # analyze attention coefficients
    if args.simulate_model == "kadapter":
        # relation: data side
        print("start to analyze relation: data side (K-Adapter)...")
        # triple statistics (P-label)
        relation_label = {}
        for d in all_data:
            tmp_r = d["relation"]
            if tmp_r in relation_label:
                tmp_count = relation_label[tmp_r]
                relation_label[tmp_r] = tmp_count + 1
            else:
                relation_label[tmp_r] = 1
        # print("Triple statistics: num of relation labels", relation_label)

        # nid to Qid
        all_idx_reverse = {v: k for k, v in all_idx.items()}
        all_qid, unlearned_qid, forgetting_qid, welllearned_qid = [], [], [], []
        for idx in self_loop_idx:
            all_qid.append(all_idx_reverse[src[idx]])
        for idx in unlearned_idx:
            unlearned_qid.append(all_idx_reverse[src[idx]])
        for idx in forgetting_idx:
            forgetting_qid.append(all_idx_reverse[src[idx]])
        for idx in welllearned_idx:
            welllearned_qid.append(all_idx_reverse[src[idx]])

        # calculate statistics
        all_qid = set(all_qid)
        unlearned_qid = set(unlearned_qid)
        forgetting_qid = set(forgetting_qid)
        welllearned_qid = set(welllearned_qid)
        print("Start to calculate the statistics... ", len(all_qid), len(unlearned_qid), len(forgetting_qid), len(welllearned_qid))
        unlearned_dist, forgetting_dist, welllearned_dist = {}, {}, {}
        for k, v in relation_label.items():
            unlearned_dist[k] = 0
            forgetting_dist[k] = 0
            welllearned_dist[k] = 0
        cross_count = 0
        for d in tqdm(all_data):
            tmp_subj_l = d["subj_label"]
            tmp_obj_l = d["obj_label"]
            tmp_rl = d["relation"]
            if tmp_subj_l in unlearned_qid or tmp_obj_l in unlearned_qid:
                unlearned_dist[tmp_rl] += 1
            elif tmp_subj_l in forgetting_qid or tmp_obj_l in forgetting_qid:
                forgetting_dist[tmp_rl] += 1
            elif tmp_subj_l in welllearned_qid or tmp_obj_l in welllearned_qid:
                welllearned_dist[tmp_rl] += 1
            else:
                cross_count += 1
        print("The distributions of triples for all, unlearned, forgetting, well-learned knowledge are: ", cross_count)
        for k, v in unlearned_dist.items(): unlearned_dist[k] = unlearned_dist[k]/relation_label[k]
        for k, v in forgetting_dist.items(): forgetting_dist[k] = forgetting_dist[k]/relation_label[k]
        for k, v in welllearned_dist.items(): welllearned_dist[k] = welllearned_dist[k]/relation_label[k]
        top5_pid = []
        tmp_top5 = sorted([v for k, v in unlearned_dist.items()], reverse=True)[5]
        for k, v in unlearned_dist.items():
            if v > tmp_top5:
                top5_pid.append(k)
        tmp_top5 = sorted([v for k, v in forgetting_dist.items()], reverse=True)[5]
        for k, v in forgetting_dist.items():
            if v > tmp_top5:
                top5_pid.append(k)
        tmp_top5 = sorted([v for k, v in welllearned_dist.items()], reverse=True)[5]
        for k, v in welllearned_dist.items():
            if v > tmp_top5:
                top5_pid.append(k)
        unlearned_dist = {k: v for k, v in unlearned_dist.items() if k in top5_pid}
        forgetting_dist = {k: v for k, v in forgetting_dist.items() if k in top5_pid}
        welllearned_dist = {k: v for k, v in welllearned_dist.items() if k in top5_pid}
        print(len(top5_pid), unlearned_dist, forgetting_dist, welllearned_dist)
        # example
        print("Start to generate examples: P275, P1313, P461...")
        unlearned_exp = []  # "P275"
        forgetting_exp = []  # "P1313"
        well_un_exp, well_fo_exp, well_well_exp = [], [], []  # "P461"
        for d in tqdm(all_data):
            if d["relation"] == "P275":
                if len(unlearned_exp) < 10:
                    if d["subj_label"] in unlearned_qid or d["obj_label"] in unlearned_qid:
                        unlearned_exp.append(d)
            elif d["relation"] == "P1313":
                if len(forgetting_exp) < 10:
                    if d["subj_label"] in forgetting_qid and d["obj_label"] in forgetting_qid:
                        forgetting_exp.append(d)
            elif d["relation"] == "P461":
                if len(well_un_exp) < 5:
                    if d["subj_label"] in unlearned_qid or d["obj_label"] in unlearned_qid:
                        well_un_exp.append(d)
                if len(well_fo_exp) < 5:
                    if d["subj_label"] in forgetting_qid and d["obj_label"] in forgetting_qid:
                        well_fo_exp.append(d)
                if len(well_well_exp) < 5:
                    if d["subj_label"] in welllearned_qid and d["obj_label"] in welllearned_qid:
                        well_well_exp.append(d)
        print(unlearned_exp)
        print(forgetting_exp)
        print(well_un_exp, well_fo_exp, well_well_exp)
    return


def task_plot(args):
    # plot_relative_score(os.path.join(args.data_dir, "results", "relative.pdf"))
    # plot random as benchmark
    # plot_random(os.path.join(args.data_dir, "results"))
    # plot_baseline(os.path.join(args.data_dir, "results", "baselines.pdf"))
    # load KG, index, attention_coefficients
    _, all_KG_nx, all_idx, all_data = load_data(args, all_data=True)
    all_attn = load_attn(args, mode="analysis")
    all_KG_nx = graph_completion(all_KG_nx)
    all_KG = dgl.from_networkx(all_KG_nx)
    all_KG = dgl.add_self_loop(all_KG)
    all_KG.edata["a"] = all_attn
    src, dst, eid = all_KG.edges(form="all")
    src = src.tolist()
    dst = dst.tolist()
    src_idx_dict = {}
    dst_idx_dict = {}
    for i in range(len(src)):
        if src[i] not in src_idx_dict:
            src_idx_dict[src[i]] = set()
        src_idx_dict[src[i]].add(i)
        if dst[i] not in dst_idx_dict:
            dst_idx_dict[dst[i]] = set()
        dst_idx_dict[dst[i]].add(i)
    # get attn & distribution
    self_attn, edge_attn = [], []
    for i in tqdm(range(len(src))):
        tmp_a = float(all_KG.edata["a"][eid[i]])
        if src[i] == dst[i]:
            self_attn.append(tmp_a)
        else:
            edge_attn.append(tmp_a)
    plot_distribution(os.path.join(args.data_dir, "results", "self_attn_distribution.pdf"), self_attn, args.simulate_model, "self-loops")
    plot_distribution(os.path.join(args.data_dir, "results", "edge_attn_distribution.pdf"), edge_attn, args.simulate_model, "edges")
    return


def task_robustness(args):  # only for K-Adapter
    # load data
    print("start to load data...")
    all_KG_nx = read_edgelist(os.path.join(args.data_dir, "all_KG.edgelist"))
    # get embedding (node features)
    print("start to load KG embedding...")
    kg_emb = deepwalk(args, all_KG_nx)
    all_KG_nx = graph_completion(all_KG_nx)
    all_KG = dgl.from_networkx(all_KG_nx)
    all_KG = dgl.add_self_loop(all_KG)
    src, dst, eid = all_KG.edges(form="all")
    # different level of noise
    gcs, lc_node, cos_node, euc_node, lc_edge, cos_edge, euc_edge = [], [], [], [], [], [], []
    for nr in range(11):
        noisy_emb = nr*np.random.normal(0, 1, kg_emb.shape) + (1-nr)*kg_emb
        # cos
        '''
        # cos_node.append(1-dist_node(noisy_emb, kg_emb, "cosine"))
        noisy_dist, kge_dist = dist_edge(noisy_emb, kg_emb, all_KG_nx, "cosine")
        print((1-kge_dist), (1-noisy_dist))
        cos_edge.append((1-kge_dist) - (1-noisy_dist))
        # euclidean similarity
        # euc_node.append(1/(1+dist_node(noisy_emb, kg_emb, "euclidean")))
        noisy_dist, kge_dist = dist_edge(noisy_emb, kg_emb, all_KG_nx, "euclidean")
        print(1/(1+kge_dist), 1/(1+noisy_dist))
        euc_edge.append(1/(1+kge_dist) - 1/(1+noisy_dist))
        # gcs
        attn_matrix = run_gcs(args, all_KG, None, kg_emb, "train", False, nr)
        self_attn = []
        all_KG.edata["a"] = attn_matrix
        for i in range(len(src)):
            if src[i] == dst[i]:
                self_attn.append(float(all_KG.edata["a"][i]))
        gcs.append(sum(self_attn)/len(self_attn))
        print(sum(self_attn)/len(self_attn))
        # linear classifier
        '''
        run_lc(args, all_KG_nx, noisy_emb, kg_emb)
    print(gcs, lc_node, cos_node, euc_node, lc_edge, cos_edge, euc_edge)
    return


def task_baselines(args):
    # load embedding
    all_KG_nx = read_edgelist(os.path.join(args.data_dir, "all_KG.edgelist"))
    all_KG_nx = graph_completion(all_KG_nx)
    if args.simulate_model == "kadapter": org_model_name = "roberta-large"
    else: org_model_name = "bert-base-uncased"
    all_feats_org = load_emb(args, None, None, org_model_name, "_all")
    all_feats_klm = load_emb(args, None, None, args.simulate_model, "_all")
    all_KG = dgl.from_networkx(all_KG_nx)
    all_KG = dgl.add_self_loop(all_KG)
    src, dst, eid = all_KG.edges(form="all")
    '''
    # RA methods: (src_list, dst_list)
    edges_cos = learned_edges(all_KG_nx, all_feats_org, all_feats_klm, "cosine")
    edges_cos = set(edges_cos)
    edges_euc = learned_edges(all_KG_nx, all_feats_org, all_feats_klm, "euclidean")
    edges_euc = set(edges_euc)
    # all_KG
    run_lc(args, all_KG_nx, all_feats_org, all_feats_klm, edges_cos, edges_euc)
    # gcs
    # all_attn = run_gcs(args, all_KG, all_feats_org, all_feats_klm, "train")
    all_attn = load_attn(args, mode="analysis")
    all_KG.edata["a"] = all_attn
    total_count, cos_count, euc_count = 0, 0, 0
    for i in tqdm(range(len(src))):
        if src[i] != dst[i]:
            if all_KG.edata["a"][i] >= args.drop_ratio:
                # learned edges
                total_count += 1
                if (int(src[i]), int(dst[i])) in edges_cos or (int(dst[i]), int(src[i])) in edges_cos:
                    cos_count += 1
                if (int(src[i]), int(dst[i])) in edges_euc or (int(dst[i]), int(src[i])) in edges_euc:
                    euc_count += 1
    print(total_count, cos_count/total_count, euc_count/total_count)
    '''
    # acc change
    all_feats_org = all_feats_org.numpy()
    all_feats_klm = all_feats_klm.numpy()
    # all_feats_org = (all_feats_org-np.mean(all_feats_org, 0))/np.std(all_feats_org, 0)
    # all_feats_klm = (all_feats_klm-np.mean(all_feats_klm, 0))/np.std(all_feats_klm, 0)
    # cos
    # cos_node.append(1-dist_node(all_feats_org, all_feats_klm, "cosine"))
    noisy_dist, kge_dist = dist_edge(all_feats_org, all_feats_klm, all_KG_nx, "cosine")
    print((1-kge_dist), (1-noisy_dist), (1-kge_dist) - (1-noisy_dist))
    # euclidean similarity
    # euc_node.append(1/(1+dist_node(all_feats_org, all_feats_klm, "euclidean")))
    noisy_dist, kge_dist = dist_edge(all_feats_org, all_feats_klm, all_KG_nx, "euclidean")
    print(1/(1+kge_dist), 1/(1+noisy_dist), 1/(1+kge_dist) - 1/(1+noisy_dist))
    # gcs
    # all_attn = run_gcs(args, all_KG, None, all_feats_klm, "train", False, nr)
    all_attn = load_attn(args, mode="analysis")
    self_attn = []
    all_KG.edata["a"] = all_attn
    for i in range(len(src)):
        if src[i] == dst[i]:
            self_attn.append(float(all_KG.edata["a"][i]))
    print(sum(self_attn)/len(self_attn))
    # linear classifier
    run_lc(args, all_KG_nx, all_feats_org, all_feats_klm)
    return


def task_downstream_results(args):
    # load dataset
    all_qids = set()
    with open(os.path.join(args.data_dir, "./OpenEntity/test.json"), "r") as f: 
        test_sets = json.loads(f.read())
    for sent in test_sets:
        for e in sent["ents"]:
            all_qids.add(e[0])
    print("Entity numbers: ", len(all_qids))
    # load attn
    _, all_KG_nx, all_idx, _ = load_data(args, all_data=True)
    all_idx = {v: k for k, v in all_idx.items()}
    all_KG_nx = graph_completion(all_KG_nx)
    all_attn = load_attn(args, mode="analysis")
    all_KG = dgl.from_networkx(all_KG_nx)
    all_KG = dgl.add_self_loop(all_KG)
    all_KG.edata["a"] = all_attn
    src, dst, _ = all_KG.edges(form="all")
    # get forgotten and learned entity Qid
    test_forgotten, test_learned = set(), set()
    for i in tqdm(range(len(src))):
        if src[i] == dst[i]:
            tmp_qid = all_idx[int(src[i])]
            if tmp_qid in all_qids:
                tmp_a = float(all_KG.edata["a"][i])
                if tmp_a > args.drop_ratio: 
                    test_learned.add(tmp_qid)  # CR & WL
                else: 
                    test_forgotten.add(tmp_qid)  # CF
    print(len(test_forgotten), len(test_learned))
    # print(test_forgotten,test_learned)
    # get w/o CF
    test_sets_forgotten, test_sets_learned = [], []
    num_count, num_hit = 0, 0
    for sent in test_sets:
        tmp_ents = sent["ents"]
        tmp_ind = 1
        for e in sent["ents"]:
            if e[0] in test_forgotten:
                # tmp_ents.remove(e)
                tmp_ind = 0
                num_hit += 1
            num_count += 1
        sent["ents"] = tmp_ents
        # print(tmp_ents)
        if tmp_ind != 0:
            test_sets_forgotten.append(sent)
        # print(test_sets_forgotten)
    print("w/o IE: ", num_count, num_hit)
    num_count, num_hit = 0, 0
    with open(os.path.join(args.data_dir, "./OpenEntity/test_noie.json"), "w") as f: 
        f.write(json.dumps(test_sets_forgotten))
    # get w/o UE
    for sent in test_sets:
        tmp_ents = sent["ents"]
        tmp_ind = 1
        for e in sent["ents"]:
            if e[0] in test_learned:
                # tmp_ents.remove(e)
                tmp_ind = 0
                num_hit += 1
            num_count += 1
        sent["ents"] = tmp_ents
        # print(tmp_ents)
        if tmp_ind != 0:
            test_sets_learned.append(sent)
        # print(test_sets_learned)
    print("w/o UE: ", num_count, num_hit)

    # get hit num
    with open(os.path.join(args.data_dir, "./OpenEntity/dev.json"), "r") as f: 
        dev_sets = json.loads(f.read())
    with open(os.path.join(args.data_dir, "./OpenEntity/train.json"), "r") as f: 
        train_sets = json.loads(f.read())
    ki_qid, dt_qid = set(), set()
    all_dataset = train_sets + dev_sets + test_sets
    for i in tqdm(range(len(src))):
        if src[i] == dst[i]:
            tmp_qid = all_idx[int(src[i])]
            ki_qid.add(tmp_qid)
    num_count, num_hit = 0, 0
    for sent in all_dataset:
        for e in sent["ents"]:
            if e[0] in ki_qid:
                num_hit += 1
            num_count += 1
    print("all num: ", num_count, num_hit)

    with open(os.path.join(args.data_dir, "./OpenEntity/test_noue.json"), "w") as f: 
        f.write(json.dumps(test_sets_learned))
    print(len(test_sets_forgotten), len(test_sets_learned))
