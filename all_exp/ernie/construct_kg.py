import json
import networkx as nx


def create_kg():
    entity2id = {}
    with open("entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)

    # for all graph
    all_id2entity = {}
    with open("data/Wikidata/knowledge_graphs/entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            all_id2entity[int(eid)] = qid

    all_subj, all_obj = [], []
    with open("data/Wikidata/knowledge_graphs/triple2id.txt") as fin:
        fin.readline()
        for line in fin:
            sid, oid, _ = line.strip().split('\t')
            all_subj.append(int(sid))
            all_obj.append(int(oid))

    # write edge_list
    g = nx.Graph()
    for i in range(len(all_subj)):
        s_qid = all_id2entity[all_subj[i]]
        o_qid = all_id2entity[all_obj[i]]
        if s_qid in entity2id and o_qid in entity2id:
            g.add_edge(entity2id[s_qid], entity2id[o_qid])

    print(len(entity2id))
    print(g.number_of_nodes(), g.number_of_edges())



if __name__ == "__main__":
    create_kg()