import os
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForPreTraining, RobertaTokenizer, RobertaModel

from kadapter.pytorch_transformers.my_modeling_roberta import RobertaModelwithAdapter
from knowledge_bert.modeling import PreTrainedBertModel, BertModel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask, labels):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.labels = labels
        self.input_ent = input_ent
        self.ent_mask = ent_mask


def convert_examples_to_features(args, tmp_text, tokenizer, entity2id, text2entity, embed, max_seq_length=256):
    inputs = tokenizer(tmp_text, return_tensors="pt", truncation=True, max_length=max_seq_length, padding="max_length")
    input_ids = inputs["input_ids"]
    input_mask = inputs["attention_mask"]
    segment_ids = inputs["token_type_ids"]

    input_ent, ent_mask = input_ids*0-1, input_ids*0
    input_ent[0] = entity2id[text2entity[tmp_text]]
    ent_mask[0] = 1

    return input_ids, input_mask, segment_ids, embed(input_ent), ent_mask


class BertForRepresentation(PreTrainedBertModel):
    def __init__(self, config, num_labels=2):
        super(BertForRepresentation, self).__init__(config)
        self.bert = BertModel(config)
        self.apply(self.init_bert_weights)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask, output_all_encoded_layers=False)
        return encoded_layers


def load_emb(args, input_text, input_idx, model_name, data_type):
    # start to calculate embedding
    emb_path = os.path.join(args.data_dir, model_name+data_type+"_emb.pt")
    if not os.path.exists(emb_path):
        if model_name == "kadapter" or model_name == "roberta-large":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            feats = torch.zeros(len(input_text), 1024)
            if model_name == "kadapter":
                pretrained_model = RobertaModelwithAdapter(args)
            else:
                pretrained_model = RobertaModel.from_pretrained(model_name)
        elif model_name == "ernie" or model_name == "bert-base-uncased":
            feats = torch.zeros(len(input_text), 768)
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            if model_name == "ernie":
                pass
                # load ernie
                # from knowledge_bert.tokenization import BertTokenizer
                # from knowledge_bert.modeling import BertForPreTraining
                # from knowledge_bert.optimization import BertAdam
                from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
                pretrained_model, _ = BertForRepresentation.from_pretrained("./ernie_thu/ernie_base", 
                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format("-1"))
            else:
                pretrained_model = BertForPreTraining.from_pretrained("bert-base-uncased", 
                                                                        return_dict=True, 
                                                                        output_hidden_states = True)
        pretrained_model = pretrained_model.to(args.device)
        pretrained_model.eval()
        print("....start to tokenize data...", model_name)
        if model_name == "ernie":
            entity2id = {}
            with open(os.path.join(args.data_dir, "entity2id.txt")) as fin:
                fin.readline()
                for line in fin:
                    qid, eid = line.strip().split('\t')
                    entity2id[qid] = int(eid)

            text2entity = {}
            with open(os.path.join(args.data_dir, "entity_map.txt")) as fin:
                for line in fin:
                    text, qid = line.strip().split('\t')
                    text2entity[text] = qid
            vecs = []
            vecs.append([0]*100)
            with open(os.path.join(args.data_dir, "entity2vec.vec"), 'r') as fin:
                for line in fin:
                    vec = line.strip().split('\t')
                    vec = [float(x) for x in vec]
                    vecs.append(vec)
            embed = torch.FloatTensor(vecs)
            embed = torch.nn.Embedding.from_pretrained(embed)
            del vecs
        for tmp_label, tmp_text in tqdm(input_text.items()):
            if model_name == "bert-base-uncased":
                inputs = tokenizer(tmp_text, return_tensors="pt").to(args.device)
                outputs = pretrained_model(**inputs)
                outputs = outputs.hidden_states[-1].detach()
            elif model_name == "ernie":
                input_ids, input_mask, segment_ids, input_ent, ent_mask = \
                            convert_examples_to_features(args, tmp_text, tokenizer, entity2id, text2entity, embed)
                input_ids = input_ids.to(args.device)
                input_mask = input_mask.to(args.device)
                segment_ids = segment_ids.to(args.device)
                input_ent = input_ent.to(args.device)
                ent_mask = ent_mask.to(args.device)
                outputs = pretrained_model(input_ids, segment_ids, input_mask, input_ent, ent_mask, None).detach()
            else:
                inputs = tokenizer(tmp_text, return_tensors="pt").to(args.device)
                outputs = pretrained_model(**inputs)[0].detach()
            # take average value of all tokens
            outputs = torch.squeeze(torch.sum(outputs, axis=1))
            feats[input_idx.get(tmp_label)] = outputs
            torch.cuda.empty_cache()
        # check error feature
        if 0 in torch.sum(feats, axis=1): 
            raise ValueError("Error value! check tokenization and pretrain model")
        # save
        torch.save(feats, emb_path)
    else:
        feats = torch.load(emb_path)

    return feats


def load_attn(args, mode="analysis"):
    if mode == "train": 
        attn = torch.load(os.path.join(args.data_dir, 
                                        "results", 
                                        str(args.temperature) + "_train_attn.pt"))
    elif mode == "test": 
        attn = torch.load(os.path.join(args.data_dir, 
                                        "results", 
                                        str(args.temperature) + "_test_attn.pt"))
    else:
        attn = torch.load(os.path.join(args.data_dir, 
                                        "results", 
                                        str(args.temperature) + "_all_attn.pt"))

    return attn


def save_attn(args, attn, mode="analysis"):
    attn = attn.detach().cpu()
    if mode == "train":
        torch.save(attn, os.path.join(args.data_dir, 
                                        "results", 
                                        str(args.temperature) + "_train_attn.pt"))
    elif mode == "test":
        torch.save(attn, os.path.join(args.data_dir, 
                                        "results", 
                                        str(args.temperature) + "_test_attn.pt"))
    else:
        torch.save(attn, os.path.join(args.data_dir, 
                                        "results", 
                                        str(args.temperature) + "_all_attn.pt"))

    return