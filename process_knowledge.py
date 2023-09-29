import os
import json
import torch
import networkx
import utils
import pickle
import numpy
from tqdm import tqdm

PPD_KG = networkx.Graph()
f = open("kg/PPD/triples.data", "r", encoding="utf-8")
for line in f:
    temp = line.strip().split(" ")
    PPD_KG.add_edge(int(temp[0][1:]), int(temp[2][1:]))

PPD_KG_embedding = torch.stack(torch.load("kg/PPD/EntityEmbedding.pt"))
#print(PPD_KG_embedding.size())

def get_subgraph(nodes):
    all_node = []
    for node in nodes:
        all_node.append(node)
        neighbors = PPD_KG.neighbors(node)
        for neighbor in neighbors:
            all_node.append(neighbor)
    all_node = list(set(all_node))
    return PPD_KG.subgraph(all_node)

#print(PPD_KG)
#print(get_subgraph([1,345,678]))

def process_knowledge_SemEval():
    data = torch.load("raw/SemEval_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        nnodes = []
        for sent in data[id]:
            try:
                nnodes += [int(a[1:]) for a in utils.IdentifyEntity(sent)]
            except:
                nnodes += []
        nnodes = list(set(nnodes))
        subgraph = get_subgraph(nnodes)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges)
        fo = [pair[0] for pair in edges]
        to = [pair[1] for pair in edges]
        nodes = [-1] + nodes # superentity of aspect K
        for node in nnodes:
            fo.append(-1)
            to.append(node)
        try:
            node_features = torch.stack([PPD_KG_embedding[i] for i in nodes[1:]]) # the randomly initialized features for supernode is ommitted here
            edge_index = torch.tensor([[nodes.index(i) for i in fo], [nodes.index(i) for i in to]]).long()
        except:
            node_features = torch.tensor([]) # to be randomly generated in the training phase
            edge_index = torch.tensor([[0],[0]]).long() #self-loop of the superentity
        # print(node_features.size())
        # print(edge_index.size())
        # print(len(nodes))
        # print(len(fo))
        # print(len(to))
        result[id] = {"node_features": node_features, "edge_index": edge_index}
    torch.save(result, "processed/SemEval_knowledge.json")

def process_knowledge_Allsides():
    data = torch.load("raw/Allsides_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        nnodes = []
        for sent in data[id]:
            try:
                nnodes += [int(a[1:]) for a in utils.IdentifyEntity(sent)]
            except:
                nnodes += []
        nnodes = list(set(nnodes))
        subgraph = get_subgraph(nnodes)
        nodes = list(subgraph.nodes)
        edges = list(subgraph.edges)
        fo = [pair[0] for pair in edges]
        to = [pair[1] for pair in edges]
        nodes = [-1] + nodes # superentity of aspect K
        for node in nnodes:
            fo.append(-1)
            to.append(node)
        try:
            node_features = torch.stack([PPD_KG_embedding[i] for i in nodes[1:]]) # the randomly initialized features for supernode is ommitted here
            edge_index = torch.tensor([[nodes.index(i) for i in fo], [nodes.index(i) for i in to]]).long()
        except:
            node_features = torch.tensor([]) # to be randomly generated in the training phase
            edge_index = torch.tensor([[0],[0]]).long() #self-loop of the superentity
        # print(node_features.size())
        # print(edge_index.size())
        # print(len(nodes))
        # print(len(fo))
        # print(len(to))
        result[id] = {"node_features": node_features, "edge_index": edge_index}
    torch.save(result, "processed/Allsides_knowledge.json")

f = open("kg/FND/entity_feature_transE.pkl", "rb")
f.seek(0)
temp = pickle.load(f)
FND_KG_embedding = torch.tensor(temp)
print(FND_KG_embedding.size())

def process_knowledge_FND():
    data = torch.load("raw/FND_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        nnodes = []
        for sent in data[id]:
            try:
                nnodes += [int(a[1:]) for a in utils.IdentifyEntityFND(sent)]
            except:
                nnodes += []
        nnodes = list(set(nnodes))
        node_features = []
        fo = []
        to = []
        i = 1
        for node in nnodes:
            node_features.append(FND_KG_embedding[node])
            fo.append(0)
            to.append(i)
            i += 1
        try:
            node_features = torch.stack(node_features)
            edge_index = torch.tensor([fo, to]).long()
        except:
            node_features = torch.tensor([]) # to be randomly generated in the training phase
            edge_index = torch.tensor([[0],[0]]).long() #self-loop of the superentity
        result[id] = {"node_features": node_features, "edge_index": edge_index}
    torch.save(result, "processed/FND_knowledge.json")

CPNET_KG_embedding = torch.tensor(numpy.load("kg/cpnet/tzw.ent.npy"))

def process_knowledge_FC():
    data = torch.load("raw/FC_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.75 * length) : int(1 * length)] # range control
    file_name = "processed/FC_knowledge4.json" # change file name
    result = {}
    for id in tqdm(ids):
        nnodes = []
        for sent in data[id]:
            try:
                nnodes += [int(a[1:]) for a in utils.IdentifyEntityCPNet(sent)]
            except:
                nnodes += []
        nnodes = list(set(nnodes))
        node_features = []
        fo = []
        to = []
        i = 1
        for node in nnodes:
            node_features.append(CPNET_KG_embedding[node])
            fo.append(0)
            to.append(i)
            i += 1
        try:
            node_features = torch.stack(node_features)
            edge_index = torch.tensor([fo, to]).long()
        except:
            node_features = torch.tensor([]) # to be randomly generated in the training phase
            edge_index = torch.tensor([[0],[0]]).long() #self-loop of the superentity
        result[id] = {"node_features": node_features, "edge_index": edge_index}
    torch.save(result, file_name)

def process_knowledge_RCVP():
    data = torch.load("raw/RCVP_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.75 * length) : int(1 * length)] # range control
    file_name = "processed/RCVP_knowledge4.json" # change file name
    result = {}
    for id in tqdm(ids):
        nnodes = []
        for sent in data[id]:
            try:
                nnodes += [int(a[1:]) for a in utils.IdentifyEntityFND(sent)]
            except:
                nnodes += []
        nnodes = list(set(nnodes))
        node_features = []
        fo = []
        to = []
        i = 1
        for node in nnodes:
            node_features.append(FND_KG_embedding[node])
            fo.append(0)
            to.append(i)
            i += 1
        try:
            node_features = torch.stack(node_features)
            edge_index = torch.tensor([fo, to]).long()
        except:
            node_features = torch.tensor([]) # to be randomly generated in the training phase
            edge_index = torch.tensor([[0],[0]]).long() #self-loop of the superentity
        result[id] = {"node_features": node_features, "edge_index": edge_index}
    torch.save(result, file_name)

# process_knowledge_SemEval()
# a = torch.load("processed/SemEval_knowledge.json")
# print(a[123]["node_features"].size())

# process_knowledge_Allsides()
# a = torch.load("processed/Allsides_knowledge.json")
# print(a[123]["node_features"].size())

# process_knowledge_FND()
# a = torch.load("processed/FND_knowledge.json")
# print(a[123]["node_features"].size())

#process_knowledge_FC()

process_knowledge_RCVP()