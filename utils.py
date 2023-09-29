import tagme
from transformers import pipeline
import transformers
import torch
import pickle
from tqdm import tqdm

tagme.GCUBE_TOKEN = "f3349d85-c9c4-452f-be7e-205df03008f9-843339462"

entity_list = []
f = open("kg/PPD/entity2id.data", "r", encoding="utf-8")
for line in f:
    entity_list.append(line.strip())
f.close()

def Annotate(txt, language="en", theta=0.1):
    annotations = tagme.annotate(txt, lang=language)
    dic = dict()
    for ann in annotations.get_annotations(theta):
        # print(ann)
        try:
            A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
            dic[(A, B)] = score
        except:
            logger.error('error annotation about ' + ann)
    return dic

# result = Annotate("President Joe Biden")
# print(result)

entity_title_list = torch.load("kg/FND/entity_title.pt")

def IdentifyEntityFND(text):
    #given text, return a list of e123s
    mentioned_entities = []
    result = Annotate(text)
    #print(result)
    for key in result:
        try:
            mentioned_entities.append("e"+str(entity_title_list.index(key[1])))
        except:
            continue
    return mentioned_entities

# result = IdentifyEntityFND("Donald Trump announces that he will run for president again.")
# print(result)

def IdentifyEntity(text):
    #given text, return a list of e123s
    mentioned_entities = []
    result = Annotate(text)
    for key in result:
        for entity in entity_list:
            if key[1].lower() in entity:
                mentioned_entities.append("e"+str(entity.strip().split(" ")[-1]))
                break
    return mentioned_entities

# result = IdentifyEntity("Senator Rick Scott blast President Biden on the reconciliation bill.")
# print(result)

entity_list_cpnet = []
f = open("kg/cpnet/concept.txt", "r", encoding="utf-8")
for line in f:
    entity_list_cpnet.append(line.strip())

def IdentifyEntityCPNet(text):
    #given text, return a list of e123s
    mentioned_entities = []
    for i in range(len(entity_list_cpnet)):
        if entity_list_cpnet[i] in text:
            mentioned_entities.append("e" + str(i))
    return mentioned_entities

# KG triple into text module
entity_text = []
f = open("kg/PPD/entity2id.data", "r", encoding="utf-8")
for line in f:
    temp = line.strip().split(" ")[:-1]
    name = temp[0]
    for i in range(1,len(temp)):
        name += (" " + temp[i])
    entity_text.append(name)
f.close()

relation_text = []
f = open("kg/PPD/relation2id.data", "r", encoding="utf-8")
for line in f:
    temp = line.strip().split(" ")[0].split("_")
    name = temp[0]
    for i in range(1,len(temp)):
        name += (" " + temp[i])
    relation_text.append(name)
f.close()

kg_adj = {}
f = open("kg/PPD/triples.data", "r", encoding="utf-8")
for line in f:
    temp = line.strip().split(" ")
    try:
        kg_adj[temp[0]].append((temp[0], temp[1], temp[2]))
    except:
        kg_adj[temp[0]] = [(temp[0], temp[1], temp[2])]
    try:
        kg_adj[temp[2]].append((temp[0], temp[1], temp[2]))
    except:
        kg_adj[temp[2]] = [(temp[0], temp[1], temp[2])]
f.close()

# def knowledge_text(entity):
#     # get text of all triples with entity
#     know_string = ""
#     for triple in kg_adj[entity]:
#         know_string += (entity_text[int(triple[0][1:])] + " " + relation_text[int(triple[1][1:])] + " " + entity_text[int(triple[2][1:])] + ". ")
#     return know_string

# # print(knowledge_text("e123"))

f = open("kg/PPD/entity_summary.txt", "r", encoding="utf-8")
entity_desc_PPD = []
for line in f:
    entity_desc_PPD.append(line.strip())

def knowledge_text(entity):
    return entity_desc_PPD[int(entity[1:])]

#print(knowledge_text("e246"))

f = open("kg/FND/entityDescCorpus.pkl", "rb")
f.seek(0)
entity_desc = pickle.load(f)
def knowledge_text_FND(entity):
    return entity_desc[int(entity[1:])]

# print(knowledge_text_FND("e7300"))

cpnet_dict = {}
f = open("kg/cpnet/conceptnet.en.csv", "r", encoding="utf-8")
for line in tqdm(f):
    temp = line.split("\t")
    try:
        cpnet_dict[temp[1]].append((temp[0],temp[2]))
    except:
        cpnet_dict[temp[1]] = [(temp[0],temp[2])]
        
def knowledge_text_CPNet(entity): # entity like "ignore"
    know_string = ""
    try:
        for tup in cpnet_dict[entity]:
            know_string += (entity + " " + tup[0] + " " + tup[1] + ". ")
    except:
        know_string = ""
    return know_string

#print(knowledge_text_CPNet("ignore"))