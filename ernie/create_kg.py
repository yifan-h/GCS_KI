import tokenization
import sys
import os
from multiprocessing import Pool
from nltk.tokenize import sent_tokenize
import math
from tqdm import tqdm
import json


vocab_file = "ernie_base/vocab.txt"
do_lower_case = True
input_folder = "pretrain_data/ann"

tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

file_list = []
for path, _, filenames in os.walk(input_folder):
    for filename in filenames:
        file_list.append(os.path.join(path, filename))

part = int(math.ceil(len(file_list) / 20.))
file_list = [file_list[i:i+part] for i in range(0, len(file_list), part)]

sep_id = tokenizer.convert_tokens_to_ids(["sepsepsep"])[0]

# load entity dict
d_ent = {}
with open("anchor2id.txt", "r") as fin:
    for line in fin:
        v = line.strip().split("\t")
        if len(v) != 2:
            continue
        d_ent[v[0]] = v[1]

used_ent = {}

def run_proc(file_list):
    folder = "pretrain_data/raw"
    for i in tqdm(range(len(file_list))):
        target = "{}/{}".format(folder, i)
        # fout_text = open(target+"_token", "w")
        # fout_ent = open(target+"_entity", "w")
        input_names = file_list[i]
        for input_name in input_names:
            # print(input_name)
            fin = open(input_name, "r")

            for doc in fin:
                doc = doc.strip()
                segs = doc.split("[_end_]")
                content = segs[0]
                sentences = sent_tokenize(content)
                map_segs = segs[1:]
                maps = {}
                for x in map_segs:
                    v = x.split("[_map_]")
                    if len(v) != 2:
                        continue
                    if v[1] in d_ent:
                        maps[v[0]] = d_ent[v[1]]
                        used_ent[v[1]] = d_ent[v[1]]
            fin.close()
        # fout_ent.close()
        # fout_text.close()

    print(len(d_ent), len(used_ent))

    with open("pretrain_data/used_entity.json", "w") as f:
        f.write(json.dumps(used_ent))

folder = "pretrain_data/raw"
if not os.path.exists(folder):
    os.makedirs(folder)

#n = int(sys.argv[1])
#p = Pool(n)
#for i in range(n):
#    p.apply_async(run_proc, args=(i,n, file_list))
#p.close()
#p.join()
run_proc(file_list)
