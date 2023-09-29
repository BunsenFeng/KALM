import torch
import torch.nn as nn
import torch_geometric
import graph_gnn_layer
import pickle
import numpy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelWithLMHead, AutoModel
from tqdm import tqdm

class InputProcess(nn.Module):
    def __init__(self, lm_name, dataset, in_channels):
        super(InputProcess, self).__init__()
        if lm_name == "deberta":
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
            self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
            self.text_dim = 768
        if lm_name == "electra":
            self.tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
            self.model = AutoModel.from_pretrained("google/electra-small-discriminator")
            self.text_dim = 256
        if lm_name == "bart":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
            self.model = AutoModel.from_pretrained("facebook/bart-base")
            self.text_dim = 768
        # if lm_name == "t5":
        #     self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        #     self.model = AutoModel.from_pretrained("t5-base")

        if dataset == "SemEval" or dataset == "Allsides":
            self.kge_dim = 768
        if dataset == "FND" or dataset == "RCVP":
            self.kge_dim = 100
        if dataset == "FC":
            self.kge_dim = 1024

        self.LinearT = nn.Linear(self.text_dim, in_channels)
        self.LinearK = nn.Linear(self.kge_dim, in_channels)
        self.LinearG = nn.Linear(self.text_dim, in_channels)
        self.activation = nn.SELU()

    # def lm_extract(self, text):
    #     inputs = self.tokenizer(text, truncation = True, padding = True, max_length = 200, return_tensors="pt")
    #     return torch.mean(self.model(inputs["input_ids"].cuda()).last_hidden_state.squeeze(0), dim=0)
    # def forward(self, data):
    #     # 6 fields of data: text, knowledge, graph, original_text, id, label
    #     text = [torch.randn(self.text_dim).cuda()] # randomly initialized supertoken
    #     for sent in data["text"]:
    #         text.append(self.lm_extract(sent))
    #     text = self.activation(self.LinearT(torch.stack(text)))

    def lm_extract(self, text):
        inputs = self.tokenizer(text, truncation = True, padding = True, max_length = 200, return_tensors="pt") # 50, 100, or 200
        return torch.mean(self.model(inputs["input_ids"].cuda()).last_hidden_state, dim=1)

    def forward(self, data):
        # 6 fields of data: text, knowledge, graph, original_text, id, label
        text = [torch.randn(1,self.text_dim).cuda()] # randomly initialized supertoken
        text = torch.cat((text[0], self.lm_extract(data["text"])), dim = 0)
        text = self.activation(self.LinearT(text))

        knowledge = {"node_features": self.activation(self.LinearK(torch.cat((torch.randn(1, self.kge_dim).cuda(), data["knowledge"]["node_features"]), dim=0))),
                     "edge_index": data["knowledge"]["edge_index"]}

        # graph = [torch.randn(self.text_dim).cuda()]
        # for sent in data["original_text"]:
        #     graph.append(self.lm_extract(sent))
        # graph = {"node_features": self.activation(self.LinearG(torch.stack(graph))), "edge_index": data["graph"]["edge_index"], "edge_type": data["graph"]["edge_type"]}

        graph = [torch.randn(1,self.text_dim).cuda()]
        graph = torch.cat((graph[0], self.lm_extract(data["original_text"])), dim = 0)
        graph = {"node_features": self.activation(self.LinearG(graph)), "edge_index": data["graph"]["edge_index"], "edge_type": data["graph"]["edge_type"]}

        return {"text": text, "knowledge": knowledge, "graph": graph, "id": data["id"], "label": data["label"]}

class LayerT(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p):
        super(LayerT, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead, batch_first=True)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input):
        result = self.dropout(self.activation(self.layer(input.unsqueeze(0)).squeeze(0)))
        return result

class LayerK(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p):
        super(LayerK, self).__init__()
        self.layer = torch_geometric.nn.GATConv(in_channels=in_channels, out_channels=int(in_channels/nhead), heads=nhead)
        #self.layer = torch_geometric.nn.GCNConv(in_channels=in_channels, out_channels=in_channels)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, input):
        result = self.dropout(self.activation(self.layer(input["node_features"], input["edge_index"])))
        return {"node_features": result,
                "edge_index": input["edge_index"]}

class LayerG(nn.Module):
    def __init__(self, in_channels, dropout_p, dataset):
        super(LayerG, self).__init__()
        if dataset == "SemEval" or dataset == "Allsides":
            KG_embedding = torch.stack(torch.load("kg/PPD/EntityEmbedding.pt"))
        if dataset == "FND" or dataset == "RCVP":
            f = open("kg/FND/entity_feature_transE.pkl", "rb")
            f.seek(0)
            temp = pickle.load(f)
            KG_embedding = torch.tensor(temp)
        if dataset == "FC":
            KG_embedding = torch.tensor(numpy.load("kg/cpnet/tzw.ent.npy"))
        num_edge_type = len(KG_embedding) + 2
        rel_dim = len(KG_embedding[0])
        rel_emb = torch.cat((torch.randn(2, rel_dim), KG_embedding), dim=0).half()

        self.layer = graph_gnn_layer.SimpleHGN(in_channels = in_channels, out_channels = in_channels, num_edge_type = num_edge_type, rel_dim = rel_dim, rel_emb = rel_emb)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        result = self.dropout(self.activation(self.layer(input["node_features"], input["edge_index"], input["edge_type"])[0]))
        return {"node_features": result,
                "edge_index": input["edge_index"],
                "edge_type": input["edge_type"]}

def att_agg(q, kv):
    # q: [d]
    # kv: [n, d]
    # return [1,d]
    return torch.matmul(torch.nn.functional.softmax(torch.matmul(q.unsqueeze(0), torch.transpose(kv,0,1)), dim=1), kv).squeeze(0)

class LayerFusion(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p):
        super(LayerFusion, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead, batch_first=True)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, Tdata, Kdata, Gdata):
        # return [6, d]
        seq = [Tdata[0]]
        seq = seq + [att_agg(seq[0],Tdata[1:])]
        seq = seq + [Kdata[0]]
        seq = seq + [att_agg(seq[0], Kdata[1:])]
        seq = seq + [Gdata[0]]
        seq = seq + [att_agg(seq[0], Gdata[1:])]
        result = self.dropout(self.activation(self.layer(torch.stack(seq).unsqueeze(0)).squeeze(0)))
        assert len(result) == 6
        return result

class RingLM(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p, dataset):
        super(RingLM, self).__init__()
        # if lm_name == "deberta":
        #     self.text_dim = 768
        # elif lm_name == "electra":
        #     self.text_dim = 256
        # if dataset == "SemEval" or dataset == "Allsides":
        #     self.kge_dim = 768
        # elif dataset == "FND":
        #     self.kge_dim = 100
        self.layerT = LayerT(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerK = LayerK(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerG = LayerG(in_channels = in_channels, dropout_p = dropout_p, dataset = dataset)
        self.layerFusion = LayerFusion(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
    
    def forward(self, input):
        newT = self.layerT(input["text"])
        newK = self.layerK(input["knowledge"])
        newG = self.layerG(input["graph"])
        result = self.layerFusion(newT, newK["node_features"], newG["node_features"])
        newT = torch.cat((result[0].unsqueeze(0), newT[1:]), dim=0)
        newK = {"node_features": torch.cat((result[2].unsqueeze(0), newK["node_features"][1:]), dim=0), 
                "edge_index": newK["edge_index"]}
        newG = {"node_features": torch.cat((result[4].unsqueeze(0), newG["node_features"][1:]), dim=0), 
                "edge_index": newG["edge_index"],
                "edge_type" : newG["edge_type"]}
        return {"text": newT, "knowledge": newK, "graph": newG, "id": input["id"], "label": input["label"]}

class RingLM_remove(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p, dataset, remove):
        super(RingLM_remove, self).__init__()
        # if lm_name == "deberta":
        #     self.text_dim = 768
        # elif lm_name == "electra":
        #     self.text_dim = 256
        # if dataset == "SemEval" or dataset == "Allsides":
        #     self.kge_dim = 768
        # elif dataset == "FND":
        #     self.kge_dim = 100
        self.layerT = LayerT(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerK = LayerK(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerG = LayerG(in_channels = in_channels, dropout_p = dropout_p, dataset = dataset)
        self.layerFusion = LayerFusion(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.remove = remove
        self.in_channels = in_channels
        if self.remove == "mint":
            self.activation = nn.GELU()
            self.dropout = nn.Dropout()
            self.linear1 = torch.nn.Sequential(self.dropout, self.activation, nn.Linear(3 * self.in_channels, 3 * self.in_channels))
            self.linear2 = torch.nn.Sequential(self.dropout, self.activation, nn.Linear(3 * self.in_channels, 3 * self.in_channels))
        if self.remove == "concat":
            self.activation = nn.GELU()
            self.dropout = nn.Dropout()
            self.linear = self.linear1 = torch.nn.Sequential(self.dropout, self.activation, nn.Linear(3 * self.in_channels, self.in_channels))
    
    def forward(self, input):
        newT = self.layerT(input["text"])
        newK = self.layerK(input["knowledge"])
        newG = self.layerG(input["graph"])
        if self.remove == "T":
            result = self.layerFusion(torch.zeros(2, self.in_channels, device = "cuda"), newK["node_features"], newG["node_features"])
        elif self.remove == "K":
            result = self.layerFusion(newT, torch.zeros(2, self.in_channels, device = "cuda"), newG["node_features"])
        elif self.remove == "G":
            result = self.layerFusion(newT, newK["node_features"], torch.zeros(2, self.in_channels, device = "cuda"))
        elif self.remove == "KG":
            result = self.layerFusion(newT, torch.zeros(2, self.in_channels, device = "cuda"), torch.zeros(2, self.in_channels, device = "cuda"))
        elif self.remove == "mint":
            result = torch.cat((newT[0].unsqueeze(0), newK["node_features"][0].unsqueeze(0)), dim = 1)
            result = torch.cat((result, newG["node_features"][0].unsqueeze(0)), dim = 1)
            result = self.linear2(self.linear1(result))
            result = [result[0, 0:1*self.in_channels], 0, result[0,1*self.in_channels:2*self.in_channels], 0, result[0,2*self.in_channels:3*self.in_channels], 0]
        elif self.remove == "concat":
            result = torch.cat((newT[0].unsqueeze(0), newK["node_features"][0].unsqueeze(0)), dim = 1)
            result = torch.cat((result, newG["node_features"][0].unsqueeze(0)), dim = 1)
            result = self.linear(result).squeeze(0)
            result = [result, 0, result, 0 , result, 0]
        elif self.remove == "sum":
            result = newT[0].unsqueeze(0) + newK["node_features"][0].unsqueeze(0) + newG["node_features"][0].unsqueeze(0)
            result = result.squeeze(0)
            result = [result, 0, result, 0 , result, 0]
        else:
            result = self.layerFusion(newT, newK["node_features"], newG["node_features"])
        newT = torch.cat((result[0].unsqueeze(0), newT[1:]), dim=0)
        newK = {"node_features": torch.cat((result[2].unsqueeze(0), newK["node_features"][1:]), dim=0), 
                "edge_index": newK["edge_index"]}
        newG = {"node_features": torch.cat((result[4].unsqueeze(0), newG["node_features"][1:]), dim=0), 
                "edge_index": newG["edge_index"],
                "edge_type" : newG["edge_type"]}
        return {"text": newT, "knowledge": newK, "graph": newG, "id": input["id"], "label": input["label"]}

class GreaseLM(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p, dataset):
        super(GreaseLM, self).__init__()
        self.in_channels = in_channels
        self.layerT = LayerT(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerK = LayerK(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.linear1 = nn.Linear(2 * in_channels, 2 * in_channels)
        self.linear2 = nn.Linear(2 * in_channels, 2 * in_channels)

    def forward(self, input):
        newT = self.layerT(input["text"])
        newK = self.layerK(input["knowledge"])
        temp = torch.cat((newT[0].unsqueeze(0), newK["node_features"][0].unsqueeze(0)), dim = 1)
        temp = self.linear2(self.linear1(temp))
        realT = torch.cat((temp[0, 0 * self.in_channels: 1 * self.in_channels].unsqueeze(0), newT[1:]), dim = 0)
        realK = torch.cat((temp[0, 1 * self.in_channels: 2 * self.in_channels].unsqueeze(0), newK["node_features"][1:]), dim = 0)
        return {"text": realT,
                "knowledge": {"node_features": realK, "edge_index": newK["edge_index"]},
                "graph": input["graph"], "id": input["id"], "label": input["label"]}

class GreaseLM2(nn.Module):
    def __init__(self, in_channels, nhead, dropout_p, dataset):
        super(GreaseLM2, self).__init__()
        self.in_channels = in_channels
        self.layerT = LayerT(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerK = LayerK(in_channels = in_channels, nhead = nhead, dropout_p = dropout_p)
        self.layerG = LayerG(in_channels = in_channels, dropout_p = dropout_p, dataset = dataset)
        self.linear1 = nn.Linear(3 * in_channels, 3 * in_channels)
        self.linear2 = nn.Linear(3 * in_channels, 3 * in_channels)

    def forward(self, input):
        newT = self.layerT(input["text"])
        newK = self.layerK(input["knowledge"])
        temp = torch.cat((newT[0].unsqueeze(0), newK["node_features"][0].unsqueeze(0)), dim = 1)
        newG = self.layerG(input["graph"])
        temp = torch.cat((temp, newG["node_features"][0].unsqueeze(0)), dim = 1)
        temp = self.linear2(self.linear1(temp))

        realT = torch.cat((temp[0, 0 * self.in_channels: 1 * self.in_channels].unsqueeze(0), newT[1:]), dim = 0)
        realK = torch.cat((temp[0, 1 * self.in_channels: 2 * self.in_channels].unsqueeze(0), newK["node_features"][1:]), dim = 0)
        realG = torch.cat((temp[0, 2 * self.in_channels: 3 * self.in_channels].unsqueeze(0), newG["node_features"][1:]), dim = 0)
        return {"text": realT,
                "knowledge": {"node_features": realK, "edge_index": newK["edge_index"]},
                "graph": {"node_features": realG, "edge_index": newG["edge_index"], "edge_type": newG["edge_type"]}, 
                "id": input["id"], "label": input["label"]}




# pre = InputProcess(lm_name = "deberta", dataset = "SemEval", in_channels = 512)
# model = RingLM(in_channels = 512, nhead = 2, dropout_p = 0.5, dataset = "SemEval")

# for i in tqdm(range(645)):
#     text = torch.load("processed/SemEval_text.json")[i]
#     knowledge = torch.load("processed/SemEval_knowledge.json")[i]
#     graph = torch.load("processed/SemEval_graph.json")[i]
#     original_text = torch.load("raw/SemEval_text.json")[i]
#     id = i
#     label = 0
#     data = {"text": text, "knowledge": knowledge, "graph": graph, "original_text": original_text, "id": id, "label": label}

#     data1 = pre(data)
#     data2 = model(model(data1))
#     assert data2["text"].size() == data1["text"].size()
#     assert data2["knowledge"]["node_features"].size() == data1["knowledge"]["node_features"].size()
#     #assert data2["knowledge"]["edge_index"] == data1["knowledge"]["edge_index"]
#     assert data2["graph"]["node_features"].size() == data1["graph"]["node_features"].size()
#     #assert data2["graph"]["edge_index"] == data1["graph"]["edge_index"]
#     #ssert data2["graph"]["edge_type"] == data1["graph"]["edge_type"]
#     assert data2["id"] == data1["id"]
#     assert data2["label"] == data1["label"]