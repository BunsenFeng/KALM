import os
import json
import torch
import networkx
import utils
import pickle
from tqdm import tqdm

#PPD_KG_embedding = torch.stack(torch.load("kg/PPD/EntityEmbedding.pt"))
#print(PPD_KG_embedding.size())

def process_graph_SemEval():
    data = torch.load("raw/SemEval_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        mentioned_entities = []
        for sent in data[id]:
            try:
                mentioned_entities.append(utils.IdentifyEntity(sent))
            except:
                mentioned_entities.append([])
        fo = []
        to = []
        edge_type = [] # 0 for dummy-sent, 1 for sent-sent, so entity relations all have +2
        for i in range(len(data[id])): # dummy-sent
            fo.append(0)
            to.append(i+1)
            edge_type.append(0)
        for i in range(1, len(data[id])): # sent-sent
            fo.append(i)
            to.append(i+1)
            edge_type.append(1)
        for i in range(len(data[id])): # knowledge coreference
            for j in range(i+1, len(data[id])):
                both = list(set(mentioned_entities[i]).intersection(set(mentioned_entities[j])))
                for entity in both:
                    fo.append(i+1) # +1 since 0 is the dummy node
                    to.append(j+1)
                    edge_type.append(int(entity[1:])+2) # +2 since the first two are dummy-sent and sent-sent
        edge_index = torch.tensor([fo, to]).long()
        edge_type = torch.tensor(edge_type).long()
        
        result[id] = {"edge_index": edge_index, "edge_type": edge_type}
    torch.save(result, "processed/SemEval_graph.json")

def process_graph_Allsides():
    data = torch.load("raw/Allsides_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        mentioned_entities = []
        for sent in data[id]:
            try:
                mentioned_entities.append(utils.IdentifyEntity(sent))
            except:
                mentioned_entities.append([])
        fo = []
        to = []
        edge_type = [] # 0 for dummy-sent, 1 for sent-sent, so entity relations all have +2
        for i in range(len(data[id])): # dummy-sent
            fo.append(0)
            to.append(i+1)
            edge_type.append(0)
        for i in range(1, len(data[id])): # sent-sent
            fo.append(i)
            to.append(i+1)
            edge_type.append(1)
        for i in range(len(data[id])): # knowledge coreference
            for j in range(i+1, len(data[id])):
                both = list(set(mentioned_entities[i]).intersection(set(mentioned_entities[j])))
                for entity in both:
                    fo.append(i+1) # +1 since 0 is the dummy node
                    to.append(j+1)
                    edge_type.append(int(entity[1:])+2) # +2 since the first two are dummy-sent and sent-sent
        edge_index = torch.tensor([fo, to]).long()
        edge_type = torch.tensor(edge_type).long()
        
        result[id] = {"edge_index": edge_index, "edge_type": edge_type}
    torch.save(result, "processed/Allsides_graph.json")

def process_graph_FND():
    data = torch.load("raw/FND_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.9 * length) : int(0.95 * length)] # range control
    file_name = "processed/FND_graph19.json" # change file name
    result = {}
    for id in tqdm(ids):
        mentioned_entities = []
        for sent in data[id]:
            try:
                mentioned_entities.append(utils.IdentifyEntityFND(sent))
            except:
                mentioned_entities.append([])
        fo = []
        to = []
        edge_type = [] # 0 for dummy-sent, 1 for sent-sent, so entity relations all have +2
        for i in range(len(data[id])): # dummy-sent
            fo.append(0)
            to.append(i+1)
            edge_type.append(0)
        for i in range(1, len(data[id])): # sent-sent
            fo.append(i)
            to.append(i+1)
            edge_type.append(1)
        for i in range(len(data[id])): # knowledge coreference
            for j in range(i+1, len(data[id])):
                both = list(set(mentioned_entities[i]).intersection(set(mentioned_entities[j])))
                for entity in both:
                    fo.append(i+1) # +1 since 0 is the dummy node
                    to.append(j+1)
                    edge_type.append(int(entity[1:])+2) # +2 since the first two are dummy-sent and sent-sent
        edge_index = torch.tensor([fo, to]).long()
        edge_type = torch.tensor(edge_type).long()
        
        result[id] = {"edge_index": edge_index, "edge_type": edge_type}
    torch.save(result, file_name)



def process_graph_FC():
    data = torch.load("raw/FC_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.25 * length) : int(0.5 * length)] # range control
    file_name = "processed/FC_graph2.json" # change file name
    result = {}
    for id in tqdm(ids):
        mentioned_entities = []
        for sent in data[id]:
            try:
                mentioned_entities.append(utils.IdentifyEntityCPNet(sent))
            except:
                mentioned_entities.append([])
        fo = []
        to = []
        edge_type = [] # 0 for dummy-sent, 1 for sent-sent, so entity relations all have +2
        for i in range(len(data[id])): # dummy-sent
            fo.append(0)
            to.append(i+1)
            edge_type.append(0)
        for i in range(1, len(data[id])): # sent-sent
            fo.append(i)
            to.append(i+1)
            edge_type.append(1)
        for i in range(len(data[id])): # knowledge coreference
            for j in range(i+1, len(data[id])):
                both = list(set(mentioned_entities[i]).intersection(set(mentioned_entities[j])))
                for entity in both:
                    fo.append(i+1) # +1 since 0 is the dummy node
                    to.append(j+1)
                    edge_type.append(int(entity[1:])+2) # +2 since the first two are dummy-sent and sent-sent
        edge_index = torch.tensor([fo, to]).long()
        edge_type = torch.tensor(edge_type).long()
        
        result[id] = {"edge_index": edge_index, "edge_type": edge_type}
    torch.save(result, file_name)

def process_graph_RCVP():
    data = torch.load("raw/RCVP_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.75 * length) : int(1 * length)] # range control
    file_name = "processed/RCVP_graph4.json" # change file name
    result = {}
    for id in tqdm(ids):
        mentioned_entities = []
        for sent in data[id]:
            try:
                mentioned_entities.append(utils.IdentifyEntityFND(sent))
            except:
                mentioned_entities.append([])
        fo = []
        to = []
        edge_type = [] # 0 for dummy-sent, 1 for sent-sent, so entity relations all have +2
        for i in range(len(data[id])): # dummy-sent
            fo.append(0)
            to.append(i+1)
            edge_type.append(0)
        for i in range(1, len(data[id])): # sent-sent
            fo.append(i)
            to.append(i+1)
            edge_type.append(1)
        for i in range(len(data[id])): # knowledge coreference
            for j in range(i+1, len(data[id])):
                both = list(set(mentioned_entities[i]).intersection(set(mentioned_entities[j])))
                for entity in both:
                    fo.append(i+1) # +1 since 0 is the dummy node
                    to.append(j+1)
                    edge_type.append(int(entity[1:])+2) # +2 since the first two are dummy-sent and sent-sent
        edge_index = torch.tensor([fo, to]).long()
        edge_type = torch.tensor(edge_type).long()
        
        result[id] = {"edge_index": edge_index, "edge_type": edge_type}
    torch.save(result, file_name)

# process_graph_SemEval()
# a = torch.load("processed/SemEval_graph.json")
# print(a[123]["edge_index"].size())

# process_graph_Allsides()
# a = torch.load("processed/Allsides_graph.json")
# print(a[123]["edge_index"].size())

#process_graph_FND()
#a = torch.load(file_name)
#print(a[123]["edge_index"].size())

#process_graph_FC()

process_graph_RCVP()