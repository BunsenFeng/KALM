import utils
import json
import torch
import os
from tqdm import tqdm
import sys

def process_text_SemEval():
    data = torch.load("raw/SemEval_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        temp = []
        for text in data[id]:
            new = text
            try:
                mentioned_entities = utils.IdentifyEntity(text)
                for entity in mentioned_entities:
                    new += (" " + utils.knowledge_text(entity)[:200])
            except:
                new = text
            new = new[:1000]
            temp.append(new)
        #print(temp)
        result[id] = temp
    torch.save(result, "processed/SemEval_text.json")  

def process_text_Allsides():
    data = torch.load("raw/Allsides_text.json")

    ids = list(data.keys())
    result = {}
    for id in tqdm(ids):
        temp = []
        for text in data[id]:
            new = text
            try:
                mentioned_entities = utils.IdentifyEntity(text)
                for entity in mentioned_entities:
                    new += (" " + utils.knowledge_text(entity)[:200])
            except:
                new = text
            new = new[:1000]
            temp.append(new)
        #print(temp)
        result[id] = temp
    torch.save(result, "processed/Allsides_text.json")

def process_text_FND():
    data = torch.load("raw/FND_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.5 * length) : int(0.55 * length)] # range control
    file_name = "processed/FND_text11.json" # change file name
    result = {}
    for id in tqdm(ids):
        temp = []
        for text in data[id]:
            new = text
            try:
                mentioned_entities = utils.IdentifyEntityFND(text)
                for entity in mentioned_entities:
                    new += (" " + utils.knowledge_text_FND(entity)[:200])
            except:
                new = text
            new = new[:1000]
            temp.append(new)
        #print(temp)
        result[id] = temp
    torch.save(result, file_name)


def process_text_FC():
    data = torch.load("raw/FC_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.75 * length) : int(1 * length)] # range control
    file_name = "processed/FC_text4.json" # change file name
    result = {}
    for id in tqdm(ids):
        temp = []
        for text in data[id]:
            new = text
            try:
                mentioned_entities = utils.IdentifyEntityCPNet(text)
                for entity in mentioned_entities:
                    new += (" " + utils.knowledge_text_CPNet(entity)[:200])
            except:
                new = text
            new = new[:1000]
            temp.append(new)
        #print(temp)
        result[id] = temp
    torch.save(result, file_name)

def process_text_RCVP():
    data = torch.load("raw/RCVP_text.json")

    ids = list(data.keys())
    length = len(ids)
    ids = ids[int(0.75 * length) : int(1 * length)] # range control
    file_name = "processed/RCVP_text4.json" # change file name
    result = {}
    for id in tqdm(ids):
        temp = []
        for text in data[id]:
            new = text
            try:
                mentioned_entities = utils.IdentifyEntityFND(text)
                for entity in mentioned_entities:
                    new += (" " + utils.knowledge_text_FND(entity)[:200])
            except:
                new = text
            new = new[:1000]
            temp.append(new)
        #print(temp)
        result[id] = temp
    torch.save(result, file_name)

# process_text_SemEval()
# a = torch.load("processed/SemEval_text.json")
# print(a[123])

# process_text_Allsides()
# a = torch.load("processed/Allsides_text.json")
# print(a[123])

#process_text_FND()
#a = torch.load(file_name)
#print(a[123])

#process_text_FC()

process_text_RCVP()
