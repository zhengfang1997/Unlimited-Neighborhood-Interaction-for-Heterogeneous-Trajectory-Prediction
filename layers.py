import math
from turtle import forward
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Callable, Optional, Tuple, Union
import ipdb
import numpy as np
from fractions import gcd
from lanegcn import *
import os
from data import ArgoDataset, collate_fn
from utils import gpu, to_long,  Optimizer, StepLR
from layers import *
import argparse


os.environ["CUDA_VISIBLE_DEVICES"] = '6,7'
device = 'cuda'




class TypeAttentionLayer(nn.Module):
    def __init__(self,idx,in_dim,out_dim,max_node):
        super(TypeAttentionLayer,self).__init__()
        # hid_dim = abs(in_dim - out_dim)
        self.idx = idx
        self.maxpool = nn.MaxPool2d((max_node//2,1),1)
        self.linear = nn.Linear(in_dim,out_dim)
        self.agg = nn.Parameter(torch.FloatTensor(2*out_dim,1))
        self.lrelu = nn.LeakyReLU()
        self.softmax0 = nn.Softmax(dim=0)
        nn.init.uniform_(self.agg.data, -1,1)

    def forward(self,inputs):
        # inputs: (n_nodes(channel_num), n_type, in_features)
        x = self.linear(inputs)
        # x: (types,objs,features)
        n_type = x.size(0)
        x = torch.cat([x, torch.stack([x[self.idx]] * n_type, dim=0)], dim=2)
        score = torch.matmul(x, self.agg).transpose(0,1)
        weights = self.lrelu(score)

        # len_w = len(weights)  # objs_num
        weights = weights.permute(1,0,2)
        # l = nn.Linear(len_w, 1).to(device)
        weights = self.maxpool(weights).squeeze(-1)  # type_num,1
        weights = self.softmax0(weights)

        return weights

class TypeAttention(nn.Module):
    # input: v,a
    # output: a
    def __init__(self, in_dim, hid_dim, n_type, max_node, nb_cnn,ablation,univ, sparse, visual):
        super(TypeAttention, self).__init__()
        self.sparse = sparse
        self.ablation = ablation
        self.univ = univ
        self.nb_cnn = nb_cnn
        self.max_node = max_node
        self.n_type = n_type
        in_node = self.max_node
        out_node = self.max_node//2
        self.linear = nn.Linear(in_node,out_node)
        self.attentions = [TypeAttentionLayer(idx,in_dim,hid_dim,max_node) for idx in range(n_type)]
        for i,attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i),attention)
        self.pl = nn.Parameter(torch.FloatTensor(1))
        self.graph_cnn1 = nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=1, padding=0)
        self.graph_cnn2 = nn.Conv2d(in_dim//2, in_dim//2, kernel_size=2, padding=0)
        self.graph_cnn3 = nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=3, padding=1)
        self.graph_cnn5 = nn.Conv2d(in_dim // 2, in_dim // 2, kernel_size=5, padding=2)
        self.visual = visual
        self.softmax = nn.Softmax(-1)


    def forward(self, V, graph,  type, ex): # graph:A , V:actor feats
        if not ex:
            return None
        V = V.squeeze(0)

        v_obs = V[0,:,:-1]
        type = V[0,:,-1].view(-1,1)
        # num_frame-1
        for length in range(len(V)-1):
            v_obs = torch.cat((v_obs,V[length+1,:,:-1]),dim=1)
        # v_obs: (n_node, n_feat * n_frame + 1) all frame's data
        v_obs = torch.cat((v_obs,type),dim=1)
        v = []
        # max_len = 0
        v_ = []
        for type in range(self.n_type):
            v.append([])
            v[type].append(v_obs[v_obs[:, -1] == (type + 1), :-1])
            v[type] = torch.cat(v[type])  # v[type]:(n_node, n_feat * n_frame)  one type's data
            # max_len = max(max_len, len(v[type]))
            pad = nn.ZeroPad2d(padding=(0, 0, 0, self.max_node - len(v[type]))).to(device)
            v[type] = pad(v[type])

            v_.append([])
            v_[type] = v[type].transpose(0,1)  # v_[type]:(n_feat * n_frame, n_node)  one type's data

            v[type] = self.linear(v_[type])
            v[type] = v[type].transpose(0, 1)
        # for type in range(self.n_type):
            # if len(v[type]) > 0:
            #     w = nn.Linear(len(v[type]), max_len).to(device)
            #     v[type] = w(v_[type])
            #     v[type] = v[type].transpose(0, 1)
            # else:
            #     v[type] = torch.zeros((v_obs.shape[1]-1, max_len)).to(device)
            #     nn.init.xavier_uniform_(v[type].data, gain=1.414)
            #     v[type] = v[type].transpose(0, 1)
        v = torch.stack(v, dim=0)
        # v: n_nodes_of_one_type * n_types * n_features

        att = []
        # super_nodes = []
        for type in range(self.n_type):
            attention = self.attentions[type](v)
            # super_nodes.append(nodes)
            att.append(attention)
        # pk = torch.stack(att).squeeze(-1)
        # gmm_pk = torch.sum(pk,dim=0)
        # gmm_pk = self.softmax0(gmm_pk)
        # for i in range(self.n_type):
        #     gmm_pk[i] = torch.sum(att[i])
        n_nodes = graph.shape[-1]
        att_mx = torch.ones((n_nodes, n_nodes)).to(device)
        # a_flag = a_flag[0, :, :]
        type = type + 1
        a_flag = np.ones((n_nodes,n_nodes))
        type1 = type.view(len(type),-1)
        type1 = type1.repeat(1, len(type))
        a_flag = type1 * 10 + type

        if self.n_type == 3:
            # 待优化：用笛卡尔乘积出现一个大的tensor 然后再用一个index矩阵直接取
            att_mx[a_flag == 1] = 1.0
            att_mx[a_flag == 11] = att[0][0]
            att_mx[a_flag == 12] = att[0][1]
            att_mx[a_flag == 13] = att[0][2]
            att_mx[a_flag == 21] = att[1][0]
            att_mx[a_flag == 22] = att[1][1]
            att_mx[a_flag == 23] = att[1][2]
            att_mx[a_flag == 31] = att[2][0]
            att_mx[a_flag == 32] = att[2][1]
            att_mx[a_flag == 33] = att[2][2]
        elif self.n_type == 2:
            # att_mx[a_flag == 1] = 1.0
            att_mx[a_flag == 11] = att[0][0]
            att_mx[a_flag == 12] = att[0][1]
            att_mx[a_flag == 21] = att[1][0]
            att_mx[a_flag == 22] = att[1][1]
            # att_mx = 1.0
            # att_mx[type[]==0,type==0] = att[0][0]
            # att_mx[a_flag == 12] = att[0][1]
            # att_mx[a_flag == 21] = att[1][0]
            # att_mx[a_flag == 22] = att[1][1]

        if self.ablation == 0:
            wei_graph = torch.mul(graph, att_mx)  # graph 1466,wei_graph 1466
        elif self.ablation == 1:
            wei_graph = self.graph_cnn1(graph)
        elif self.ablation == 5:
            wei_graph =self.graph_cnn5(graph)
        # 1. full
        # wei_graph = torch.mul(graph, att_mx) #graph 1466,wei_graph 1466
        #
        # 2. linear instead typeatt
        # wei_graph = self.graph_cnn1(graph)

        # 3. cnn instead typeatt
        # wei_graph = self.graph_cnn5(graph)

        if self.nb_cnn:
            graph_pad = nn.ZeroPad2d(padding=(0, self.max_node - len(graph), 0, self.max_node - len(graph))).to(device)
            wei_graph2 = graph_pad(wei_graph)
            # wei_graph3 = graph_pad(wei_graph)
            # wei_graph3 = wei_graph2
            for i in range(wei_graph2.shape[-1] - graph.shape[-1]):
                wei_graph2 = self.graph_cnn2(wei_graph2)
            # for j in range(wei_graph3.shape[-1] - graph.shape[-1]-1):
            #     wei_graph2 = self.graph_cnn5(wei_graph2)

        # ablation study

        if self.univ == 2:
            graph = self.pl * wei_graph2 + graph  # wei_graph2 1466 graph 1466
        elif self.univ == 1:
            graph = self.graph_cnn1(graph)
        elif self.univ ==5:
            graph == self.graph_cnn5(graph)
        # 1. full
        # graph = self.pl * wei_graph2  + graph # wei_graph2 1466 graph 1466
        #
        # 2. linear instead uniconv
        # graph = self.graph_cnn1(graph)
        #
        # 3. cnn instead univconv
        # graph = self.graph_cnn5(graph)

        'attention visualization'
        if self.visual:
            vis_att = self.softmax(att_mx)
            vis_ta = np.zeros((len(att), len(att)))
            for i in range(len(att)):
                vis_ta[i, :] = att[i].transpose(0, 1).cpu().detach().numpy()
            attn_v = np.array(vis_att.detach().cpu().numpy())
            attn_f = np.array(a_flag.detach().cpu().numpy())
            v_speed0 = np.array(V[0].detach().cpu().numpy())
            v_speed1 = np.array(V[1].detach().cpu().numpy())
            v_speed2 = np.array(V[2].detach().cpu().numpy())
            v_speed3 = np.array(V[3].detach().cpu().numpy())
            np.savetxt('./vis1/attn_v' + str(attn_v[1, 1]) + '.txt', attn_v)
            np.savetxt('./vis1/attn_f' + str(attn_v[1, 1]) + '.txt', attn_f)
            np.savetxt('./vis1/attn_c' + str(attn_v[1, 1]) + '.txt', vis_ta)
            np.savetxt('./vis1/vspeed0' + str(attn_v[1, 1]) + '.txt', v_speed0)
            np.savetxt('./vis1/vspeed1' + str(attn_v[1, 1]) + '.txt', v_speed1)
            np.savetxt('./vis1/vspeed2' + str(attn_v[1, 1]) + '.txt', v_speed2)
            np.savetxt('./vis1/vspeed3' + str(attn_v[1, 1]) + '.txt', v_speed3)
            print(str(attn_v[1, 1]))

        if self.sparse:
            # graph = F.dropout(graph, 0.4, training=True, inplace=False)
            # graph = zero_sf(graph, -2)
            graph = F.relu(graph)

        return graph

def actor_gather(actors):
    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [x.transpose(1, 2) for x in actors] # shape: n_actors_of_one_scene, 3, 20frames
    actors = torch.cat(actors, 0) # shape: n_actors_of_all_scenes, 3, 20frames

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)  # list of all actors' id in scenes
        count += num_actors[i]
    return actors, actor_idcs

def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(
            graphs[i]["feats"].device
        )
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    return graph

class ActorAttention(nn.Module):
    def __init__(self, dim_in = 128, dim_hid = 128, n_heads=4):
        super(ActorAttention, self).__init__()
        self.config = config
        self.n_heads = n_heads
        self.dim_hid = dim_hid
        assert dim_hid % n_heads == 0
        self.dim_in = dim_in
        self.conv_q = nn.Conv2d(dim_in, dim_hid//n_heads, kernel_size=1)
        self.conv_k = nn.Conv2d(dim_in, dim_hid//n_heads, kernel_size=1)
        self.conv_v = nn.Conv2d(dim_in, dim_hid//n_heads, kernel_size=1)
        # self.norm_factor = 1 / math.sqrt(dim_k)
        self.scale = 1 / torch.sqrt(torch.FloatTensor([dim_hid // n_heads]))
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(dim_hid, dim_hid)

        norm = "GN"
        ng = 1
        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]
        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=1))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=1))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        n = 128
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors):
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])
        actors = out
        # actors:23,128,20  chw
        out = self.output(out)[:, :, -1]
        # out: 23,128
        x = actors.permute(1,0,2) # 128,23,20
        dim, n_actors, seq_len = x.shape
        q = self.conv_q(x).permute(2,1,0).view(seq_len, n_actors, self.n_heads, dim//self.n_heads) # 20,23,4,32
        k = self.conv_k(x).permute(2,1,0).view(seq_len, n_actors, self.n_heads, dim//self.n_heads)
        v = self.conv_v(x).permute(2,1,0).view(seq_len, n_actors, self.n_heads, dim//self.n_heads)
        energy = torch.matmul(q.permute(0,2,1,3), k.permute(0,2,3,1)) * self.scale # 20, 4, 23, 23
        edges = torch.mean(energy, dim=1).squeeze(1)
        mask = edges>=0.5
        edges = edges.masked_fill(mask,0)
        edges = zero_sf(edges, dim=-1)
        scores = torch.softmax(energy, dim=-1)
        att = torch.matmul(scores, v.permute(0,2,1,3)).permute(0,2,1,3) #20,23,4,32
        att = att.view(seq_len,n_actors,dim)
        att = self.linear(att)
        return edges, att, out

def zero_sf(x, dim=0, eps=1e-5):
    x_exp = torch.pow(torch.exp(x) - 1, exponent=2)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    x = x_exp / (x_exp_sum + eps)
    return x

class ActorSTGCN(nn.Module):
    def __init__(self,dim_in=128, dim_out=128, t_dim_in=3, t_dim_hid=32, t_dim_out=128, kernel_size=1, t_kernel_size=1):
        super(ActorSTGCN, self).__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size)
        self.tcn = nn.ModuleList()
        t_dim = [32, 64, 128]
        for i in range(len(t_dim)):
            if i == 0:
                self.tcn.extend([nn.Conv2d(t_dim_in, t_dim_hid, t_kernel_size),
                                 nn.GroupNorm(1, t_dim_hid),
                                 nn.PReLU()])
            else:
                self.tcn.extend([
                    nn.Conv2d(t_dim[i-1], t_dim[i], t_kernel_size),
                    nn.GroupNorm(1, t_dim[i]),
                    nn.PReLU()])
        self.norm = nn.GroupNorm(1,128)
        self.downsample = nn.Conv2d()

    # t:output[0], x:output[2], A：edge
    def forward(self, t, x, A):
        x = self.conv(x)
        h = torch.bmm(x,A)
        h = self.norm(h)
        t = self.tcn(t)
        x += t
        x = self.norm(x)
        x = self.downsample(x)
        return x


class SparseLaneAttention(nn.Module):
    def __init__(self, dim_in=128, dim_k=128, dim_v=128, conv=True):
        super(SparseLaneAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.l_q = nn.Linear(dim_in, dim_k, bias=False)
        self.l_k = nn.Linear(dim_in, dim_k, bias=False)
        self.l_v = nn.Linear(dim_in, dim_v, bias=False)
        self.norm_factor = 1 / math.sqrt(dim_k)
        self.relu = nn.ReLU(inplace=True)

        self.conv = conv

        if self.conv:
            self.lane_embed = nn.Sequential(
                nn.Conv2d(2,128,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128,128,1),
                nn.GroupNorm(1,128),
                nn.ReLU(inplace=True)
            )
            self.displ_embed = nn.Sequential(
                nn.Conv2d(2, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 1),
                nn.GroupNorm(1, 128),
                nn.ReLU(inplace=True)
            )
        else:
            self.lane_embed = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(inplace=True),
                Linear(128,128, norm='GN', ng=1, act=False),
            )
            self.displ_embed = nn.Sequential(
                nn.Linear(2, 128),
                nn.ReLU(inplace=True),
                Linear(128,128, norm='GN', ng=1, act=False),
            )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))
        fuse = dict()
        for key in keys:
            fuse[key] = []
        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(1, 128))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(128, 128, norm=norm, act=False))
                else:
                    fuse[key].append(nn.Linear(128, 128, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)

    def forward(self, graph, ex):
        if not ex:
            return None
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)
        q = feat
        feat += self.seg(graph["feats"])
        k = feat
        v = self.seg(graph["feats"])

        q = graph['ctrs']
        k = graph['feats']
        v = graph['feats']
        q = self.l_q(q)
        k = self.l_k(k)
        v = self.l_v(v)
        energy = torch.bmm(q, k.transose(1,2)) * self._norm_factor
        scores = torch.softmax(energy, dim=-1)
        att = torch.bmm(scores, v)

        feat += att
        # feat = self.relu(feat)
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]

class UnlimitedInteractionLayer(nn.Module):
    def __init__(self, actor_size, lane_size, t_size=20, dim_in=128, dim_out=128):
        super(UnlimitedInteractionLayer, self).__init__()
        self.actor_size = actor_size
        self.w_size = actor_size + t_size + lane_size
        self.conv =  nn.Conv2d(dim_in, dim_out, 1)

        self.A = torch.randn(self.w_size, self.w_size)
        self.w = nn.Conv2d(self.w_size, self.actor_size, 1)

        self.weight = nn.Parameter(torch.Tensor(dim_in, dim_out))
        self.bias = nn.Parameter(torch.Tensor(dim_out))
        self.reset_parameters()
        # self.normalize()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def normalize(self, A):
        A = A >= 0.5
        d = torch.sum(A,dim=1)
        D = torch.diag(d)
        D = torch.sqrt(D) / D
        A = torch.matmul(D, torch.matmul(A,D))
        return A

    def forward(self, actor, lane):
        V = torch.cat((actor,lane),0)
        A = self.normalize(self.A)
        V = self.conv(V.premute(1,0)).permute(0,1)
        out = torch.matmul(A, V)
        return A,V

class UnlimitedInteraction(nn.Module):
    def __init__(self, actor_size, lane_size, t_size=20, dim_in=128, dim_out=128,max_pool=True):
        super.__init__(UnlimitedInteraction, self).__init__()
        self.UI = nn.Sequential(
            UnlimitedInteractionLayer(actor_size, lane_size, t_size=20, dim_in=128, dim_out=256),
            UnlimitedInteractionLayer(actor_size, lane_size, t_size=20, dim_in=256, dim_out=256)
        )
        self.max_pool = max_pool
        if self.max_pool == True:
            self.pool = nn.MaxPool2d((1,128))
        else:
            self.pool = nn.LSTM()
        self.relu = nn.ReLU()

    def forward(self, actor, lane):
        x=self.relu(self.UI(actor, lane))
        return F.log_softmax(x, dim=1)
