import torch
import torch.nn as nn
import torch_geometric
import graph_gnn_layer
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead, AutoModel
from tqdm import tqdm

def pad_collate(x):
    return x

class KGAPDataset(Dataset):
    def __init__(self, dataset, fold, tdt, part):
        self.part = part
        self.dataset = dataset
        self.fold = fold
        self.graph = torch.load("processed/" + dataset + "_KGAP.json")
        label = torch.load("raw/" + dataset + "_label.pt")

        # legislator representation and special input format for RCVP dataset
        if dataset == "RCVP":
            self.leg_rep = torch.load("raw/learned_reps.pt", map_location=torch.device("cpu"))
            self.input = torch.load("raw/RCVP_input.json")

        # self.label processing
        self.label = {}
        if dataset == "SemEval":
            values = ["False", "True"]
            for id in label.keys():
                self.label[int(id)] = values.index(label[id])
        elif dataset == "Allsides":
            values = ["Left", "Center", "Right"]
            for id in label.keys():
                try:
                    self.label[int(id)] = values.index(label[id])
                except:
                    continue
        elif dataset == "FND":
            for id in label.keys():
                self.label[int(id)] = label[id] - 1 # -1 since 1,2,3,4
        elif dataset == "RCVP":
            for id in label.keys():
                self.label[int(id)] = label[id]


        # self.id processing
        self.id = []
        if dataset == "SemEval":
            folds = torch.load("raw/SemEval_fold.json")
            if tdt == "train":
                for i in range(10):
                    if i == fold:
                        continue
                    self.id = self.id + folds[i]
            elif tdt == "dev" or tdt == "test":
                self.id = folds[fold]
        elif dataset == "Allsides":
            folds = torch.load("raw/Allsides_fold.json")
            if tdt == "train":
                self.id = folds[fold]["train"]
            elif tdt == "dev" or tdt == "test":
                self.id = folds[fold]["dev"]
        elif dataset == "FND":
            folds = torch.load("raw/FND_fold.json")
            self.id = folds[fold][tdt]
        elif dataset == "RCVP":
            folds = torch.load("raw/RCVP_fold.json")
            self.id = folds[fold][tdt]

        if self.dataset == "FND" and self.fold == 0:
            random.shuffle(self.id)

        if tdt == "train":
            self.id = self.id[:int(part * len(self.id))]

        random.shuffle(self.id)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        if self.dataset == "RCVP":
            leg_rep = []
            for leg in self.input[self.id[index]]["legislator"]:
                leg_rep.append(self.leg_rep[leg])
            leg_rep = torch.stack(leg_rep, dim=0)
            graph = self.graph[self.input[self.id[index]]["bill"]]
            graph["node_features"] = torch.randn(2, 768)
            return {"graph": graph,
                    "id": self.id[index],
                    "label": self.label[self.id[index]],
                    "leg_rep": leg_rep}
        else:
            return {"graph": self.graph[self.id[index]],
                    "id": self.id[index],
                    "label": self.label[self.id[index]]}

def get_dataloaders_KGAP(dataset, fold, batch_size, part):
    train_data = KGAPDataset(dataset, fold, "train", part)
    dev_data = KGAPDataset(dataset, fold, "dev", part)
    test_data = KGAPDataset(dataset, fold, "test", part)

    train_loader = DataLoader(train_data, batch_size = batch_size, collate_fn = pad_collate, num_workers = 4)
    dev_loader = DataLoader(dev_data, batch_size = len(dev_data), collate_fn = pad_collate, num_workers = 4)
    test_loader = DataLoader(test_data, batch_size = len(test_data), collate_fn = pad_collate, num_workers = 4)

    return train_loader, dev_loader, test_loader

class RingLMDataset(Dataset):
    def __init__(self, dataset, fold, tdt):
        # dataset: SemEval, Allsides, FND, FC, RCVP
        # fold: 0-9 for SemEval, 0-2 for Allsides, 0-1 for FND, 0-2 for FC, 0-1 for RCVP
        self.dataset = dataset
        self.text = torch.load("processed/" + dataset + "_text.json")
        self.knowledge = torch.load("processed/" + dataset + "_knowledge.json")
        self.graph = torch.load("processed/" + dataset + "_graph.json")
        self.original_text = torch.load("raw/" + dataset + "_text.json")
        label = torch.load("raw/" + dataset + "_label.pt")

        # summary for FC dataset
        if dataset == "FC":
            self.summary = torch.load("processed/FC_summary.json")

        # legislator representation and special input format for RCVP dataset
        if dataset == "RCVP":
            self.leg_rep = torch.load("raw/learned_reps.pt", map_location=torch.device("cpu"))
            self.input = torch.load("raw/RCVP_input.json")
        
        # self.label processing
        self.label = {}
        if dataset == "SemEval":
            values = ["False", "True"]
            for id in label.keys():
                self.label[int(id)] = values.index(label[id])
        elif dataset == "Allsides":
            values = ["Left", "Center", "Right"]
            for id in label.keys():
                try:
                    self.label[int(id)] = values.index(label[id])
                except:
                    continue
        elif dataset == "FND":
            for id in label.keys():
                self.label[int(id)] = label[id] - 1 # -1 since 1,2,3,4
                # if fold == 0 and tdt == "dev":
                #     self.label[int(id)] = 1 - self.label[int(id)] # fold 0 dev set 0-1 flip;
                # if fold == 1 and tdt == "test":
                #     if self.label[int(id)] == 0:
                #         self.label[int(id)] = 1
                #     elif self.label[int(id)] == 1:
                #         self.label[int(id)] = 0 # fold 1 test set 0-1 flip, 2 3 remain the same
        elif dataset == "FC" or dataset == "RCVP":
            for id in label.keys():
                self.label[int(id)] = label[id]



        # self.id processing
        self.id = []
        if dataset == "SemEval":
            folds = torch.load("raw/SemEval_fold.json")
            if tdt == "train":
                for i in range(10):
                    if i == fold:
                        continue
                    self.id = self.id + folds[i]
            elif tdt == "dev" or tdt == "test":
                self.id = folds[fold]
        elif dataset == "Allsides":
            folds = torch.load("raw/Allsides_fold.json")
            if tdt == "train":
                self.id = folds[fold]["train"]
            elif tdt == "dev" or tdt == "test":
                self.id = folds[fold]["dev"]
        elif dataset == "FND":
            folds = torch.load("raw/FND_fold.json")
            self.id = folds[fold][tdt]
        elif dataset == "FC":
            folds = torch.load("raw/FC_fold.json")
            self.id = folds[fold][tdt]
        elif dataset == "RCVP":
            folds = torch.load("raw/RCVP_fold.json")
            self.id = folds[fold][tdt]

        random.shuffle(self.id)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        if self.dataset == "RCVP":
            knowledge = {"node_features": self.knowledge[self.input[self.id[index]]["bill"]]["node_features"].half(), "edge_index": self.knowledge[self.input[self.id[index]]["bill"]]["edge_index"]}
        else:
            knowledge = {"node_features": self.knowledge[self.id[index]]["node_features"].half(), "edge_index": self.knowledge[self.id[index]]["edge_index"]}
        if self.dataset == "FC":
            return {"text": self.text[self.id[index]],
                "knowledge": knowledge,
                "graph": self.graph[self.id[index]],
                "original_text": self.original_text[self.id[index]],
                "summary": self.summary[self.id[index]],
                "id": self.id[index],
                "label": self.label[self.id[index]]}
        elif self.dataset == "SemEval" or self.dataset == "Allsides" or self.dataset == "FND":
            return {"text": self.text[self.id[index]],
                "knowledge": knowledge,
                "graph": self.graph[self.id[index]],
                "original_text": self.original_text[self.id[index]],
                "id": self.id[index],
                "label": self.label[self.id[index]]}
        elif self.dataset == "RCVP":
            leg_rep = []
            for leg in self.input[self.id[index]]["legislator"]:
                leg_rep.append(self.leg_rep[leg])
            leg_rep = torch.stack(leg_rep, dim=0)
            return {"text": self.text[self.input[self.id[index]]["bill"]],
                    "knowledge": knowledge,
                    "graph": self.graph[self.input[self.id[index]]["bill"]],
                    "original_text": self.original_text[self.input[self.id[index]]["bill"]],
                    "bill_id": self.input[self.id[index]]["bill"],
                    # "leg_rep": self.leg_rep[self.input[self.id[index]]["legislator"]],
                    "leg_rep": leg_rep,
                    "id": self.id[index],
                    "label": self.label[self.id[index]]
            }

def get_dataloaders(dataset, fold, batch_size):
    train_data = RingLMDataset(dataset, fold, "train")
    dev_data = RingLMDataset(dataset, fold, "dev")
    test_data = RingLMDataset(dataset, fold, "test")

    train_loader = DataLoader(train_data, batch_size = batch_size, collate_fn = pad_collate, num_workers = 4)
    dev_loader = DataLoader(dev_data, batch_size = len(dev_data), collate_fn = pad_collate, num_workers = 4)
    test_loader = DataLoader(test_data, batch_size = len(test_data), collate_fn = pad_collate, num_workers = 4)
    
    if dataset == "RCVP":
        return train_loader, test_loader, test_loader # only temporarily
    else:
        return train_loader, dev_loader, test_loader

# a, b, c = get_dataloaders("RCVP", 0, 32)
# for batch in c:
#     for sample in batch:
#         print(sample["original_text"])
#         print(sample["leg_rep"])
#         print(sample["id"])
#         exit()
#         continue

class RingLMDataset_trainpart(Dataset):
    def __init__(self, dataset, fold, tdt, part):
        # dataset: SemEval, Allsides, FND, FC, RCVP
        # fold: 0-9 for SemEval, 0-2 for Allsides, 0-1 for FND, 0-2 for FC, 0-1 for RCVP
        self.dataset = dataset
        self.fold = fold
        self.text = torch.load("processed/" + dataset + "_text.json")
        self.knowledge = torch.load("processed/" + dataset + "_knowledge.json")
        self.graph = torch.load("processed/" + dataset + "_graph.json")
        self.original_text = torch.load("raw/" + dataset + "_text.json")
        label = torch.load("raw/" + dataset + "_label.pt")

        # summary for FC dataset
        if dataset == "FC":
            self.summary = torch.load("processed/FC_summary.json")

        # legislator representation and special input format for RCVP dataset
        if dataset == "RCVP":
            self.leg_rep = torch.load("raw/learned_reps.pt", map_location=torch.device("cpu"))
            self.input = torch.load("raw/RCVP_input.json")
        
        # self.label processing
        self.label = {}
        if dataset == "SemEval":
            values = ["False", "True"]
            for id in label.keys():
                self.label[int(id)] = values.index(label[id])
        elif dataset == "Allsides":
            values = ["Left", "Center", "Right"]
            for id in label.keys():
                try:
                    self.label[int(id)] = values.index(label[id])
                except:
                    continue
        elif dataset == "FND":
            for id in label.keys():
                self.label[int(id)] = label[id] - 1 # -1 since 1,2,3,4
                # if fold == 0 and tdt == "dev":
                #     self.label[int(id)] = 1 - self.label[int(id)] # fold 0 dev set 0-1 flip;
                # if fold == 1 and tdt == "test":
                #     if self.label[int(id)] == 0:
                #         self.label[int(id)] = 1
                #     elif self.label[int(id)] == 1:
                #         self.label[int(id)] = 0 # fold 1 test set 0-1 flip, 2 3 remain the same
        elif dataset == "FC" or dataset == "RCVP":
            for id in label.keys():
                self.label[int(id)] = label[id]



        # self.id processing
        self.id = []
        if dataset == "SemEval":
            folds = torch.load("raw/SemEval_fold.json")
            if tdt == "train":
                for i in range(10):
                    if i == fold:
                        continue
                    self.id = self.id + folds[i]
            elif tdt == "dev" or tdt == "test":
                self.id = folds[fold]
        elif dataset == "Allsides":
            folds = torch.load("raw/Allsides_fold.json")
            if tdt == "train":
                self.id = folds[fold]["train"]
            elif tdt == "dev" or tdt == "test":
                self.id = folds[fold]["dev"]
        elif dataset == "FND":
            folds = torch.load("raw/FND_fold.json")
            self.id = folds[fold][tdt]
        elif dataset == "FC":
            folds = torch.load("raw/FC_fold.json")
            self.id = folds[fold][tdt]
        elif dataset == "RCVP":
            folds = torch.load("raw/RCVP_fold.json")
            self.id = folds[fold][tdt]

        if self.dataset == "FND" and self.fold == 0:
            random.shuffle(self.id)

        if tdt == "train":
            self.id = self.id[:int(part * len(self.id))] # part = 0.1, 0.2, ..., 0.9, 1

        random.shuffle(self.id)

    def __len__(self):
        return len(self.id)

    def __getitem__(self, index):
        if self.dataset == "RCVP":
            knowledge = {"node_features": self.knowledge[self.input[self.id[index]]["bill"]]["node_features"].half(), "edge_index": self.knowledge[self.input[self.id[index]]["bill"]]["edge_index"]}
        else:
            knowledge = {"node_features": self.knowledge[self.id[index]]["node_features"].half(), "edge_index": self.knowledge[self.id[index]]["edge_index"]}
        if self.dataset == "FC":
            return {"text": self.text[self.id[index]],
                "knowledge": knowledge,
                "graph": self.graph[self.id[index]],
                "original_text": self.original_text[self.id[index]],
                "summary": self.summary[self.id[index]],
                "id": self.id[index],
                "label": self.label[self.id[index]]}
        elif self.dataset == "SemEval" or self.dataset == "Allsides" or self.dataset == "FND":
            return {"text": self.text[self.id[index]],
                "knowledge": knowledge,
                "graph": self.graph[self.id[index]],
                "original_text": self.original_text[self.id[index]],
                "id": self.id[index],
                "label": self.label[self.id[index]]}
        elif self.dataset == "RCVP":
            leg_rep = []
            for leg in self.input[self.id[index]]["legislator"]:
                leg_rep.append(self.leg_rep[leg])
            leg_rep = torch.stack(leg_rep, dim=0)
            return {"text": self.text[self.input[self.id[index]]["bill"]],
                    "knowledge": knowledge,
                    "graph": self.graph[self.input[self.id[index]]["bill"]],
                    "original_text": self.original_text[self.input[self.id[index]]["bill"]],
                    "bill_id": self.input[self.id[index]]["bill"],
                    # "leg_rep": self.leg_rep[self.input[self.id[index]]["legislator"]],
                    "leg_rep": leg_rep,
                    "id": self.id[index],
                    "label": self.label[self.id[index]]
            }

def get_dataloaders_trainpart(dataset, fold, batch_size, part):
    train_data = RingLMDataset_trainpart(dataset, fold, "train", part)
    dev_data = RingLMDataset_trainpart(dataset, fold, "dev", part)
    test_data = RingLMDataset_trainpart(dataset, fold, "test", part)

    train_loader = DataLoader(train_data, batch_size = batch_size, collate_fn = pad_collate, num_workers = 4)
    dev_loader = DataLoader(dev_data, batch_size = len(dev_data), collate_fn = pad_collate, num_workers = 4)
    test_loader = DataLoader(test_data, batch_size = len(test_data), collate_fn = pad_collate, num_workers = 4)
    
    if dataset == "RCVP":
        return train_loader, test_loader, test_loader # only temporarily
    else:
        return train_loader, dev_loader, test_loader