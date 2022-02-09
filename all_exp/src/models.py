import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax


class linear_classifier(nn.Module):
    def __init__(self, feat_dim=0, mlp=False):
        super(linear_classifier, self).__init__()
        self.linear_fh = nn.Linear(feat_dim*2, 1)
        self.mlp = mlp

    def forward(self, src, dst):
        # layers_num = 0
        h = torch.cat((src, dst), dim=1)
        h = self.linear_fh(h)
        if self.mlp: h = torch.sigmoid(h)
        return torch.squeeze(h)


class gcs_layer(nn.Module):
    def __init__(self, g, dim_num, num_heads, temperature, attn_drop):
        super(gcs_layer, self).__init__()
        self.g = g
        self.num_heads = num_heads
        self.temperature = temperature
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_q = nn.Parameter(torch.Tensor(size=(dim_num, num_heads * 64)))
        self.attn_k = nn.Parameter(torch.Tensor(size=(dim_num, num_heads * 64)))
        self.iv = nn.Linear(dim_num, dim_num)
        nn.init.xavier_normal_(self.iv.weight.data)
        nn.init.constant_(self.attn_q.data, 1)
        nn.init.constant_(self.attn_k.data, 1)

    def forward(self, h):
        v = self.iv(h)
        q = torch.matmul(h, self.attn_q).reshape((h.shape[0], self.num_heads, -1))
        k = torch.matmul(h, self.attn_k).reshape((h.shape[0], self.num_heads, -1))
        self.g.ndata.update({"q" : q, "k" : k, "v" : v})
        # calculate attention
        self.g.apply_edges(self.edge_attention)
        # attention softmax (edge-wise)
        self.g.edata["a"] = edge_softmax(self.g, self.g.edata["a"])
        # attention dropout
        self.g.edata["a"] = self.attn_drop(self.g.edata["a"])
        # message passing
        self.g.update_all(fn.src_mul_edge("v", "a", "m"), fn.sum("m", "v"))

        return self.g.ndata["v"], self.g.edata["a"]

    def edge_attention(self, edges):
        e_q = edges.src["q"]
        e_k = edges.dst["k"]
        a = e_q * e_k / 8
        a = torch.mean(a, dim=1)  # multi-head
        a = torch.mean(a, dim=1)  # scalar
        # normalization
        m = a.mean(0, keepdim=True)
        s = a.std(0, unbiased=False, keepdim=True)
        a = a - m
        a = a / s
        # temperature
        a = a / self.temperature
        # a = torch.sigmoid(a / self.temperature)

        return {"a": a}


class gcs_attention(nn.Module):
    def __init__(self, g, dim_num, num_heads, temperature, mlp_drop, attn_drop):
        super(gcs_attention, self).__init__()
        self.mlp_drop = nn.Dropout(mlp_drop)
        self.bijection_tf = nn.Linear(dim_num, dim_num)
        self.g_convolution = gcs_layer(g, 
                                        dim_num, 
                                        num_heads,
                                        temperature,
                                        attn_drop)
        self.bijection_ft = nn.Linear(dim_num, dim_num)
        self.mi_func = nn.Sequential(nn.Linear(2 * dim_num, 64),
                                    nn.Dropout(mlp_drop),
                                    nn.ELU(),
                                    nn.Linear(64, 1),
                                    nn.Dropout(mlp_drop))
        self.loss_fcn = torch.nn.L1Loss()

    def forward(self, feats_org, feats_klm):
        # GFT
        feats_org = self.mlp_drop(feats_org)  # dropout
        h = self.bijection_tf(feats_org)
        h = F.elu(h)
        # GC
        h, attn = self.g_convolution(h)
        # RGFT
        h = self.mlp_drop(h)  # dropout
        h = self.bijection_ft(h)
        h = F.elu(h)
        return h, attn

    def mi_loss(self, feats_org, feats_klm):
        h, attn = self.forward(feats_org, feats_klm)
        # calculate MI loss
        random_index = torch.randperm(feats_klm.shape[0])
        shuffle_klm = feats_klm[random_index]
        T0 = self.mi_func(torch.cat([feats_org, feats_klm], dim=-1))
        T1 = self.mi_func(torch.cat([feats_org, shuffle_klm], dim=-1))
        lower_bound = T0.mean() - torch.log(T1.exp().mean())
        return - lower_bound, attn

    def rc_loss(self, feats_org, feats_klm):
        h, attn = self.forward(feats_org, feats_klm)
        # calculate reconstruction loss
        return self.loss_fcn(h, feats_klm), attn