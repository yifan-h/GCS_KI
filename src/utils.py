import os
import json
import random
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from gensim.models import Word2Vec
import csrgraph as cg
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cosine


def learned_edges(G, emb_org, emb_klm, dist_name):
    # learned edges
    emb_org = emb_org.numpy()
    emb_klm = emb_klm.numpy()
    emb_org = (emb_org-np.mean(emb_org, 0))/np.std(emb_org, 0)
    emb_klm = (emb_klm-np.mean(emb_klm, 0))/np.std(emb_klm, 0)
    G.remove_edges_from(nx.selfloop_edges(G))
    edge_list = []
    for (src, dst) in tqdm(G.edges()):
        if dist_name == "cosine":
            org_sim = 1-cosine(emb_org[src], emb_org[dst])
            klm_sim = 1-cosine(emb_klm[src], emb_klm[dst])
            if klm_sim > org_sim:
                edge_list.append((src, dst))
        else:
            org_sim = 1/(1+euclidean(emb_org[src], emb_org[dst]))
            klm_sim = 1/(1+euclidean(emb_klm[src], emb_klm[dst]))
            if klm_sim > org_sim:
                edge_list.append((src, dst))
    print(len(edge_list))
    # print(src_list, dst_list)
    return edge_list


def binarize2np(pt):
    return (pt>0.5).float().numpy()


def dist_node(noisy_emb, kg_emb, dist_name):
    dists = []
    for i in tqdm(range(noisy_emb.shape[0])):
        if dist_name == "cosine":
            dist = cosine(noisy_emb[i], kg_emb[i])
        else:
            dist = euclidean(noisy_emb[i], kg_emb[i])
        dists.append(dist)
    return sum(dists)/len(dists)


def dist_edge(noisy_emb, kg_emb, G, dist_name):
    G.remove_edges_from(nx.selfloop_edges(G))
    noisy_dists, kge_dist = [], []
    for (src, dst) in tqdm(G.edges()):
        if dist_name == "cosine":
            ndist = cosine(noisy_emb[src], noisy_emb[dst])
            gdist = cosine(kg_emb[src], kg_emb[dst])
        else:
            ndist = euclidean(noisy_emb[src], noisy_emb[dst])
            gdist = euclidean(kg_emb[src], kg_emb[dst])
        noisy_dists.append(ndist)
        kge_dist.append(gdist)
    return sum(noisy_dists)/len(noisy_dists), sum(kge_dist)/len(kge_dist)


def get_negative_samples(src_idx, dst_idx, edge_label, negative_sample_num, all_KG_nx):
    tmp_count = 0
    for sid in src_idx:
        for did in dst_idx:
            if (sid, did) not in all_KG_nx.edges():
                src_idx.append(sid)
                dst_idx.append(did)
                edge_label.append(0)
                tmp_count += 1
            if tmp_count > negative_sample_num:
                return src_idx, dst_idx, edge_label


def get_edge_idx(src_idx, dst_idx, all_KG_nx):
    negative_sample_num = 2
    ns_num = int(negative_sample_num*len(src_idx))
    # construct edge label list
    edge_label = [1 for i in range(len(src_idx))]
    # get negative samples
    src_idx, dst_idx, edge_label = get_negative_samples(src_idx, dst_idx, edge_label, ns_num, all_KG_nx)
    # shuffle
    new_idx = [i for i in range(len(src_idx))]
    random.shuffle(new_idx)
    new_src_idx = [src_idx[i] for i in new_idx]
    new_dst_idx = [dst_idx[i] for i in new_idx]
    new_edge_label = [edge_label[i] for i in new_idx]

    return new_src_idx, new_dst_idx, new_edge_label


def plot_scatter(path, x, y, c):
    # settings
    figure_size = 4
    figure(figsize=((figure_size*(1+np.sqrt(5))), figure_size*1.8))
    mpl.rcParams['font.size'] = 25
    # plot data
    plt.plot(x, y, "o", markersize=0.5, rasterized=True)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='yellow')
    plt.text(0.3, 800, "Pearson Correlation Coefficient: "+str(c)[:7], fontsize=20, fontweight="bold")
    # plot axis
    plt.ylabel("# of Aligned Sentences", fontweight="bold")
    plt.xlabel("Attention Coefficient", fontweight="bold")
    # plt.title("Pearson correlation coefficient: ", fontweight="bold")
    # save figure
    plt.savefig(path, format='pdf')


def plot_distribution(path, a, simulate_model, type_name):
    # settings
    figure_size = 6
    figure(figsize=((figure_size*(1+np.sqrt(5))), figure_size*1.2))
    mpl.rcParams['font.size'] = 38
    # construct cmap
    # plot data
    # sns.distplot(a, hist=True, kde=False, color="tab:blue")
    sns.histplot(a, stat='probability', color="tab:blue", bins=50)
    avg_a = sum(a)/len(a)
    plt.axvline(avg_a, color='k', linestyle='--')
    # sns.distplot(a)
    # plot axis
    if simulate_model == "kadapter": 
        model_text = "K-Adapter"
    else:
        model_text = "ERNIE"
    if type_name == "self-loops":
        yl = "Ratio of Entities"
    else:
        yl = "Ratio of Triples"
    ax = plt.gca()
    plt.ticklabel_format(style='sci', axis='y', useOffset=False)
    plt.text(0.53, 0.8, model_text + ": " + type_name, fontsize=30, transform=ax.transAxes, fontweight="bold")
    plt.ylabel(yl, fontweight="bold")
    plt.xlabel("Attention Coefficient", fontweight="bold")
    # plt.legend(loc='upper right', prop={'size': 42})
    # plt.title("Pearson correlation coefficient: ", fontweight="bold")
    # save figure
    plt.savefig(path, format='pdf', bbox_inches="tight")


def plot_downstream(path):
    # settings
    figure_size = 6
    figure(figsize=((figure_size*(1+np.sqrt(5))), figure_size*1.65))
    mpl.rcParams['font.size'] = 38
    fig, ax = plt.subplots()
    # data
    num_list = [0.5052316891, 0.7318982387, 0.7475570033, 0.7559055118, 0.7594064653, 0.7641470467]
    name_list = ["ERNIE", "K-Adapter"]
    plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list)

def plot_baseline(path):
    # data
    x_list = [i*0.1 for i in range(11)]
    cos_sim = [1.15E-10,0.1581089102,0.1519965111,0.148658337,0.1498960246,0.1474460825,0.1462295256,0.142758754,0.141954903,0.1397238916,0.1427667238]
    euc_sim = [-1.73E-12,0.0798565817,0.1089288076,0.118796959,0.1238936059,0.1268398882,0.1288041509,0.1302225356,0.131295659,0.1321060244,0.1328077766]
    lin_gap = [-0.003758414534,0.215239796,0.1587141511,0.1084400991,0.1098136423,0.1324143039,0.09769262093,0.1311911656,0.1175883642,0.1085386383,0.1186916838]  # AUC score
    gcs_self = [0.269111939,0.6837203147,0.6100049789,0.6405633276,0.7121831736,0.6946454633,0.67707198,0.6995747854,0.6712743949,0.6491522599,0.7278647611]  # RCLoss
    cos_ernie = 0.2432360849
    cos_kadapter = -0.2579222328
    euc_ernie = -0.01658272111
    euc_kadapter = -0.001745202487
    lin_ernie = 0.0178529659
    lin_kadapter = -0.002959180498
    gcs_ernie = 0.4922704365
    gcs_kadapter = 0.5548360168
    # setting
    figure_size = 6
    figure(figsize=((figure_size*(1+np.sqrt(5))), figure_size*1.8))
    mpl.rcParams['font.size'] = 38
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.plot(x_list, cos_sim, "go-", label="Cosine similarity", linewidth=2)
    plt.plot(x_list, [cos_ernie for _ in range(len(x_list))], "g--", label="CS - ERNIE", linewidth=2)
    plt.plot(x_list, [cos_kadapter for _ in range(len(x_list))], "g:", label="CS - K-Adapter", linewidth=2)
    plt.plot(x_list, euc_sim, "yo-", label="Euclidean similarity", linewidth=2)
    plt.plot(x_list, [euc_ernie for _ in range(len(x_list))], "y--", label="ES - ERNIE", linewidth=2)
    plt.plot(x_list, [euc_kadapter for _ in range(len(x_list))], "y:", label="ES - K-Adapter", linewidth=2)
    plt.plot(x_list, lin_gap, "bo-", label="Linear classifier - AUC", linewidth=2)
    plt.plot(x_list, [lin_ernie for _ in range(len(x_list))], "b--", label="LC - ERNIE", linewidth=2)
    plt.plot(x_list, [lin_kadapter for _ in range(len(x_list))], "b:", label="LC - K-Adapter", linewidth=2)
    plt.plot(x_list, gcs_self, "ro-", label="GCS - integration score", linewidth=2)
    plt.plot(x_list, [gcs_ernie for _ in range(len(x_list))], "r--", label="GCS - ERNIE", linewidth=2)
    plt.plot(x_list, [gcs_kadapter for _ in range(len(x_list))], "r:", label="GCS - K-Adapter", linewidth=2)
    # plt.ylabel(, fontweight="bold")
    plt.xlabel("Ratio of noise on the input KGE", fontweight="bold")
    # plt.legend(loc='center left', prop={'size': 38})#, 'weight':'bold'})
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.8])
    lgd = ax.legend(bbox_to_anchor=(0.5, 1.66), loc=9, ncol=2, prop={'size': 32})   
    # plt.legend(bbox_to_anchor=(0, 1.01), loc="center", borderaxespad=0.1, prop={'size': 38})
    # plt.title("Pearson correlation coefficient: ", fontweight="bold")
    # save figure
    plt.savefig(path, format='pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_random(path):
    # data
    num_list = [5565478, 1091152, 561687, 127728, 0]
    y_labels = ["P-Micro", "R-Micro", "F1-Micro", "P-Macro", "R-Macro", "F1-Macro"]
    gcs_00 = [0.7663101604, 0.7526260504, 0.7594064653, 0.6297130464, 0.6237904571, 0.6267377601]
    gcs_001 = [0.7655246253, 0.7510504202, 0.7582184517, 0.6289622956, 0.6248748749, 0.6269119229]
    gcs_01 = [0.7594537815, 0.7594537815, 0.7594537815, 0.6364698031, 0.6307140474, 0.6335788535]
    gcs_09 = [0.7653390743, 0.7468487395, 0.7559808612, 0.6261261261, 0.6209542876, 0.6235294827]
    gcs_10 = [0.7736165456, 0.7268907563, 0.7495261305, 0.6206206206, 0.6054387721, 0.6129357007]
    random_00 = gcs_00
    random_001 = [0.7631296892, 0.7478991597, 0.7554376658, 0.6260427094, 0.6213713714, 0.6236982937]
    random_01 = [0.7614431879, 0.7426470588, 0.7519276788, 0.6128628629, 0.6141975309, 0.613529471]
    random_09 = [0.7765957447, 0.7284663866, 0.7517615176, 0.6172839506, 0.6049382716, 0.6110487592]
    random_10 = gcs_10
    # plot
    for i in range(len(y_labels)):
        # settings
        figure_size = 6
        figure(figsize=((figure_size*(1+np.sqrt(5))), figure_size*1.8))
        mpl.rcParams['font.size'] = 38
        # plot
        y_l = y_labels[i]
        gcs_data, random_data = [], []
        gcs_data = [gcs_00[i], gcs_001[i], gcs_01[i], gcs_09[i], gcs_10[i]]
        random_data = [random_00[i], random_001[i], random_01[i], random_09[i], random_10[i]]
        plt.plot(num_list, gcs_data, 'r--', label='GCS')
        plt.plot(num_list, gcs_data, 'ro')
        plt.plot(num_list, random_data, 'k--', label='Random')
        plt.plot(num_list, random_data, 'k*')
        plt.fill_between(num_list, random_data, gcs_data, color="tab:blue")
        # plot axis
        plt.ylabel(y_l, fontweight="bold")
        plt.xlabel("# of Sentences for Integration", fontweight="bold")
        plt.legend(loc='lower right', prop={'size': 42, 'weight':'bold'})
        # plt.title("Pearson correlation coefficient: ", fontweight="bold")
        # save figure
        plt.savefig(os.path.join(path, y_l+".pdf"), format='pdf')


def graph_completion(g):
    maxid = 0
    for nid in g.nodes():
        if maxid < nid: maxid = nid
    for i in range(maxid):
        if i not in g.nodes():
            g.add_node(i)
    return g


def graph_construction(data):
    # json to networkx
    name2id_dict = {}  # label: id
    count = 0
    KG = nx.Graph()
    for d in data:
        subj_label = d.get("subj_label")
        obj_label = d.get("obj_label")
        if subj_label not in name2id_dict:
            name2id_dict[subj_label] = count
            count += 1
        if obj_label not in name2id_dict:
            name2id_dict[obj_label] = count
            count += 1
        KG.add_edge(name2id_dict[subj_label], name2id_dict[obj_label])

    return KG, name2id_dict


def graph_construction_wiki(args, subset=None):
    # label2id
    entity2id, id2entity= {}, {}
    with open(os.path.join(args.data_dir, "entity2id.txt")) as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)
            id2entity[int(eid)] = qid
    # label2text
    entity2text = {}
    with open(os.path.join(args.data_dir, "entity_map.txt")) as fin:
        for line in fin:
            text, qid = line.strip().split('\t')
            entity2text[qid] = text
    # remove nodes in entity2id but not in entitiy2text
    qid_remove_set, eid_remove_set = set(), set()
    for k, v in entity2id.items():
        if k not in entity2text:
            qid_remove_set.add(k)
            eid_remove_set.add(v)
    for rmqid in qid_remove_set:
        entity2id.pop(rmqid)
    for rmeid in eid_remove_set:
        id2entity.pop(rmeid)

    # for all graph
    all_id2entity = {}
    with open(os.path.join(args.data_dir, "all_entity2id.txt")) as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            all_id2entity[int(eid)] = qid
    all_subj, all_obj = [], []
    with open(os.path.join(args.data_dir, "all_triple2id.txt")) as fin:
        fin.readline()
        for line in fin:
            sid, oid, _ = line.strip().split('\t')
            all_subj.append(int(sid))
            all_obj.append(int(oid))

    # get edge list
    g = nx.Graph()
    for i in range(len(all_subj)):
        s_qid = all_id2entity[all_subj[i]]
        o_qid = all_id2entity[all_obj[i]]
        if s_qid in entity2id and o_qid in entity2id:
            g.add_edge(entity2id[s_qid], entity2id[o_qid])

    if subset is None:
        for k, v in entity2id.items(): g.add_node(int(v))
        new_g = nx.Graph()
        new_entity2id = {}
        new_entity2text = {}
        count = 0
        for _, nid in entity2id.items():
            if nid in id2entity:
                new_entity2id[id2entity[nid]] = count
                count += 1
            else:
                print("warning: unexceptional nid in subset!")
        for qid in new_entity2id:
            if qid not in new_entity2text:
                if qid in entity2text:
                    new_entity2text[qid] = entity2text[qid]
        for i in range(len(all_subj)):
            s_qid = all_id2entity[all_subj[i]]
            o_qid = all_id2entity[all_obj[i]]
            if s_qid in new_entity2id and o_qid in new_entity2id:
                new_g.add_edge(new_entity2id[s_qid], new_entity2id[o_qid])
    else:  # rearrange KG id and get new entity2id dict
        new_g = nx.Graph()
        new_entity2id = {}
        new_entity2text = {}
        count = 0
        for nid in subset:
            if nid in id2entity:
                new_entity2id[id2entity[nid]] = count
                count += 1
            else:
                print("warning: unexceptional nid in subset!")
        for qid in new_entity2id:
            if qid not in new_entity2text:
                if qid in entity2text:
                    new_entity2text[qid] = entity2text[qid]
        for i in range(len(all_subj)):
            s_qid = all_id2entity[all_subj[i]]
            o_qid = all_id2entity[all_obj[i]]
            if s_qid in new_entity2id and o_qid in new_entity2id:
                new_g.add_edge(new_entity2id[s_qid], new_entity2id[o_qid])

    return new_g, new_entity2id, new_entity2text


def graph_split(G):
    # pick first large connected component
    remove_set = set()
    count = 0  # keep the first largest connected component
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        if count == 0: 
            subgraph_set = c
        else:
            remove_set = remove_set.union(c)
        count += 1
    G.remove_nodes_from(remove_set)

    # splite subgraph into two components (degree distribution)
    degree_list = list(G.degree())
    degree_list.sort(key=lambda tup:tup[1], reverse=True)
    train_set, test_set = set(), set()
    for (nid, deg) in degree_list:
        if nid not in train_set and nid not in test_set:
            if random.random() > len(train_set) / (len(train_set)+len(test_set)+1):
                train_set.add(nid)
            else:
                test_set.add(nid)

    # remove isolated nodes
    subgraph_train = G.subgraph(train_set)
    subgraph_test = G.subgraph(test_set)
    train_remove, test_remove = [], []
    for nid in train_set:
        if subgraph_train.degree(nid) == 0:
            train_remove.append(int(nid))
    for nid in train_remove: train_set.remove(nid)
    for nid in test_set:
        if subgraph_test.degree(nid) == 0:
            test_remove.append(int(nid))
    for nid in test_remove: test_set.remove(nid)

    return train_set, test_set


def read_edgelist(path):
    g = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            tmp_list = line.split(' ')
            src_id = int(tmp_list[0])
            dst_id = int(tmp_list[1])
            g.add_edge(src_id, dst_id)
    return g


def deepwalk(args, G, num_walks=10, walk_len=40, string_nid=False):
    if os.path.exists(os.path.join(args.data_dir, "kg_emb.npy")):
        return np.load(os.path.join(args.data_dir, "kg_emb.npy"))
    # add self loop
    for nid in G.nodes(): G.add_edge(nid, nid)
    kgemb_idx = [str(idx) for idx in range(G.number_of_nodes())]
    # walking
    G = cg.csrgraph(G, threads=20)
    walks = G.random_walks(walklen=walk_len, 
                epochs=num_walks, 
                start_nodes=None,
                return_weight=1.,
                neighbor_weight=1.)
    paths = []
    for i in tqdm(range(walks.shape[0])):
        tmp_path = []
        for j in range(walks.shape[1]):
            tmp_path.append(str(walks[i][j]))
        paths.append(tmp_path)
    '''
    if not string_nid:
        for nid in tqdm(G.nodes()):
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [str(nid)]
                for j in range(walk_len):
                    neighbors = [str(n) for n in G.neighbors(int(tmp_path[-1]))]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)
    else:
        for nid in tqdm(G.nodes()):
            if G.degree(nid) == 0: continue
            for i in range(num_walks):
                tmp_path = [nid]
                for j in range(walk_len):
                    neighbors = [n for n in G.neighbors(tmp_path[-1])]
                    tmp_path.append(random.choice(neighbors))
                paths.append(tmp_path)
    '''
    kgemb_model = Word2Vec(paths, 
                            vector_size=128, 
                            window=5, 
                            min_count=0, 
                            sg=1, 
                            hs=1, 
                            workers=20)
    kgemb_vec = kgemb_model.wv[kgemb_idx]
    np.save(os.path.join(args.data_dir, "kg_emb.npy"), kgemb_vec)
    return kgemb_vec


def load_data(args, all_data=False):
    if args.simulate_model == "ernie":
        entity2id_path = os.path.join(args.data_dir, "entity2id.txt")
        all_entity2id_path = os.path.join(args.data_dir, "all_entity2id.txt")
        all_triple2id_path = os.path.join(args.data_dir, "all_triple2id.txt")
        if all_data:  # analysis
            # load data
            if not os.path.exists(os.path.join(args.data_dir, "all_KG.edgelist")) \
                or not os.path.exists(os.path.join(args.data_dir, "all_idx.json")):
                if os.path.exists(entity2id_path) \
                    and os.path.exists(all_entity2id_path) \
                    and os.path.exists(all_triple2id_path):
                    all_KG, all_idx, all_text = graph_construction_wiki(args)
                else:
                    raise ValueError("Error! check data path and readme file for ernie pretrain data.")
                print("....start to construct KG...")
                # save KG and idx
                all_KG, all_idx, all_text = graph_construction_wiki(args)
                nx.write_edgelist(all_KG, os.path.join(args.data_dir, "all_KG.edgelist"))
                with open(os.path.join(args.data_dir, "all_idx.json"), "w") as f: 
                    f.write(json.dumps(all_idx))
                with open(os.path.join(args.data_dir, "all_text.json"), "w") as f: 
                    f.write(json.dumps(all_text))
            else:
                all_KG = read_edgelist(os.path.join(args.data_dir, "all_KG.edgelist"))
                with open(os.path.join(args.data_dir, "all_idx.json"), "r") as f: 
                    all_idx = json.loads(f.read())
                with open(os.path.join(args.data_dir, "all_text.json"), "r") as f: 
                    all_text = json.loads(f.read())
            print("....all_KG: # n, # e: ", all_KG.number_of_nodes(), all_KG.number_of_edges())
            return all_text, all_KG, all_idx, all_data
        else:  # training & test
            # load data
            if not os.path.exists(os.path.join(args.data_dir, "train_KG.edgelist")) \
                or not os.path.exists(os.path.join(args.data_dir, "test_KG.edgelist"))\
                or not os.path.exists(os.path.join(args.data_dir, "train_idx.json"))\
                or not os.path.exists(os.path.join(args.data_dir, "test_idx.json")):
                # split data
                print("....start to split KG...")
                if os.path.exists(entity2id_path) \
                    and os.path.exists(all_entity2id_path) \
                    and os.path.exists(all_triple2id_path):
                    KG, idx_dict, text_dict = graph_construction_wiki(args)
                    print("........KG: ", KG.number_of_nodes(), KG.number_of_edges())
                    # graph split (degree distribution)
                    train_set, test_set = graph_split(KG)
                else:
                    raise ValueError("Error! check data path and readme file for ernie pretrain data.")
                # save
                print("....start to construct KG...")
                train_KG, train_idx, train_text = graph_construction_wiki(args, train_set)
                test_KG, test_idx, test_text = graph_construction_wiki(args, test_set)
                nx.write_edgelist(train_KG, os.path.join(args.data_dir, "train_KG.edgelist"))
                nx.write_edgelist(test_KG, os.path.join(args.data_dir, "test_KG.edgelist"))
                with open(os.path.join(args.data_dir, "train_idx.json"), "w") as f: 
                    f.write(json.dumps(train_idx))
                with open(os.path.join(args.data_dir, "test_idx.json"), "w") as f: 
                    f.write(json.dumps(test_idx))
                with open(os.path.join(args.data_dir, "train_text.json"), "w") as f: 
                    f.write(json.dumps(train_text))
                with open(os.path.join(args.data_dir, "test_text.json"), "w") as f: 
                    f.write(json.dumps(test_text))
            else:
                train_KG = read_edgelist(os.path.join(args.data_dir, "train_KG.edgelist"))
                test_KG = read_edgelist(os.path.join(args.data_dir, "test_KG.edgelist"))
                with open(os.path.join(args.data_dir, "train_idx.json"), "r") as f: 
                    train_idx = json.loads(f.read())
                with open(os.path.join(args.data_dir, "test_idx.json"), "r") as f: 
                    test_idx = json.loads(f.read())
                with open(os.path.join(args.data_dir, "train_text.json"), "r") as f: 
                    train_text = json.loads(f.read())
                with open(os.path.join(args.data_dir, "test_text.json"), "r") as f: 
                    test_text = json.loads(f.read())
            print("....Train_KG: # n, # e: ", train_KG.number_of_nodes(), train_KG.number_of_edges())
            print("....Test_KG: # n, # e: ", test_KG.number_of_nodes(), test_KG.number_of_edges())
            return train_text, train_KG, train_idx, test_text, test_KG, test_idx
    else:  # k-adapter
        if all_data:  # analysis
            all_feats_path = os.path.join(args.data_dir, "all.json")
            # load data
            if not os.path.exists(all_feats_path)\
                or not os.path.exists(os.path.join(args.data_dir, "all_KG.edgelist")) \
                or not os.path.exists(os.path.join(args.data_dir, "all_idx.json")):
                if os.path.exists(all_feats_path):
                    with open(all_feats_path, "r") as f: 
                        all_data = json.loads(f.read())
                    all_KG, all_idx = graph_construction(all_data)
                else:
                    raise ValueError("Error! check data path or run k-adpater to preprocess it")
                print("....start to construct KG...")
                # construct clean data
                all_text = {}  # label: text
                for d in all_data:
                    subj_label = d.get("subj_label")
                    obj_label = d.get("obj_label")
                    tmp_tokens = d.get("token")
                    if subj_label not in all_text:
                        tmp_text = tmp_tokens[d.get("subj_start"):d.get("subj_end")+1]
                        tmp_text = ''.join(tmp_text)
                        all_text[subj_label] = tmp_text
                    if obj_label not in all_text:
                        tmp_text = tmp_tokens[d.get("obj_start"):d.get("obj_end")+1]
                        tmp_text = ''.join(tmp_text)
                        all_text[obj_label] = tmp_text
                # save KG and idx
                nx.write_edgelist(all_KG, os.path.join(args.data_dir, "all_KG.edgelist"))
                with open(os.path.join(args.data_dir, "all_idx.json"), "w") as f: 
                    f.write(json.dumps(all_idx))
                with open(os.path.join(args.data_dir, "all_text.json"), "w") as f: 
                    f.write(json.dumps(all_text))
            else:
                with open(all_feats_path, "r") as f: all_data = json.loads(f.read())
                all_KG = read_edgelist(os.path.join(args.data_dir, "all_KG.edgelist"))
                with open(os.path.join(args.data_dir, "all_idx.json"), "r") as f: 
                    all_idx = json.loads(f.read())
                with open(os.path.join(args.data_dir, "all_text.json"), "r") as f: 
                    all_text = json.loads(f.read())
            print("....All_KG: # n, # e: ", all_KG.number_of_nodes(), all_KG.number_of_edges())
            return all_text, all_KG, all_idx, all_data
        else:  # training & test
            # set path
            train_feats_path = os.path.join(args.data_dir, "train.json")
            test_feats_path = os.path.join(args.data_dir, "test.json")
            # load data
            if not os.path.exists(train_feats_path) or not os.path.exists(test_feats_path)\
                or not os.path.exists(os.path.join(args.data_dir, "train_KG.edgelist")) \
                or not os.path.exists(os.path.join(args.data_dir, "test_KG.edgelist"))\
                or not os.path.exists(os.path.join(args.data_dir, "train_idx.json"))\
                or not os.path.exists(os.path.join(args.data_dir, "test_idx.json")):
                # split data
                print("....start to split KG...")
                all_feats_path = os.path.join(args.data_dir, "all.json")
                if os.path.exists(all_feats_path):
                    with open(all_feats_path, "r") as f: data = json.loads(f.read())
                    KG, idx_dict = graph_construction(data)
                    # graph split (degree distribution)
                    train_set, test_set = graph_split(KG)
                else:
                    raise ValueError("Error! check data path or run k-adpater to preprocess it")
                # save data
                train_data, test_data = [], []
                for d in data:
                    subj_label = d.get("subj_label")
                    obj_label = d.get("obj_label")
                    if idx_dict[subj_label] in train_set and idx_dict[obj_label] in train_set:
                        train_data.append(d)
                    if idx_dict[subj_label] in test_set and idx_dict[obj_label] in test_set:
                        test_data.append(d)
                with open(train_feats_path, "w") as f: f.write(json.dumps(train_data))
                with open(test_feats_path, "w") as f: f.write(json.dumps(test_data))
                # construct two graphs (train & test)
                print("....start to construct KG...")
                train_KG, train_idx = graph_construction(train_data)  # label: id
                test_KG, test_idx = graph_construction(test_data)
                # construct clean data
                train_text, test_text = {}, {}  # label: text
                for d in train_data:
                    subj_label = d.get("subj_label")
                    obj_label = d.get("obj_label")
                    tmp_tokens = d.get("token")
                    if subj_label not in train_text:
                        tmp_text = tmp_tokens[d.get("subj_start"):d.get("subj_end")+1]
                        tmp_text = ''.join(tmp_text)
                        train_text[subj_label] = tmp_text
                    if obj_label not in train_text:
                        tmp_text = tmp_tokens[d.get("obj_start"):d.get("obj_end")+1]
                        tmp_text = ''.join(tmp_text)
                        train_text[obj_label] = tmp_text
                for d in test_data:
                    subj_label = d.get("subj_label")
                    obj_label = d.get("obj_label")
                    tmp_tokens = d.get("token")
                    if subj_label not in test_text:
                        tmp_text = tmp_tokens[d.get("subj_start"):d.get("subj_end")+1]
                        tmp_text = ''.join(tmp_text)
                        test_text[subj_label] = tmp_text
                    if obj_label not in test_text:
                        tmp_text = tmp_tokens[d.get("obj_start"):d.get("obj_end")+1]
                        tmp_text = ''.join(tmp_text)
                        test_text[obj_label] = tmp_text
                # save KG and idx
                nx.write_edgelist(train_KG, os.path.join(args.data_dir, "train_KG.edgelist"))
                nx.write_edgelist(test_KG, os.path.join(args.data_dir, "test_KG.edgelist"))
                with open(os.path.join(args.data_dir, "train_idx.json"), "w") as f: 
                    f.write(json.dumps(train_idx))
                with open(os.path.join(args.data_dir, "test_idx.json"), "w") as f: 
                    f.write(json.dumps(test_idx))
                with open(os.path.join(args.data_dir, "train_text.json"), "w") as f: 
                    f.write(json.dumps(train_text))
                with open(os.path.join(args.data_dir, "test_text.json"), "w") as f: 
                    f.write(json.dumps(test_text))
            else:
                train_KG = read_edgelist(os.path.join(args.data_dir, "train_KG.edgelist"))
                test_KG = read_edgelist(os.path.join(args.data_dir, "test_KG.edgelist"))
                with open(os.path.join(args.data_dir, "train_idx.json"), "r") as f: 
                    train_idx = json.loads(f.read())
                with open(os.path.join(args.data_dir, "test_idx.json"), "r") as f: 
                    test_idx = json.loads(f.read())
                with open(os.path.join(args.data_dir, "train_text.json"), "r") as f: 
                    train_text = json.loads(f.read())
                with open(os.path.join(args.data_dir, "test_text.json"), "r") as f: 
                    test_text = json.loads(f.read())
            print("....Train_KG: # n, # e: ", train_KG.number_of_nodes(), train_KG.number_of_edges())
            print("....Test_KG: # n, # e: ", test_KG.number_of_nodes(), test_KG.number_of_edges())
            return train_text, train_KG, train_idx, test_text, test_KG, test_idx
