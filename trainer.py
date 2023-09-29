import sys
import model
import torch
import torch.nn as nn
import dataloader
from tqdm import tqdm
import os
import pytorch_lightning as pl
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime

class RingLM_trainer(pl.LightningModule):
    def __init__(self, dataset, fold, hyperparams):
        super().__init__()
        self.dataset = dataset
        self.fold = fold
        self.optimizer = hyperparams["optimizer"]
        self.lm_name = hyperparams["lm_name"]
        self.in_channels = hyperparams["in_channels"]
        self.lr_lm = hyperparams["lr_lm"]
        self.lr = hyperparams["lr"]
        self.weight_decay = hyperparams["weight_decay"]
        self.lm_freeze_epochs = hyperparams["lm_freeze_epochs"]
        self.log_dir = hyperparams["log_dir"]
        if self.dataset == "SemEval":
            self.num_class = 2
        elif self.dataset == "Allsides":
            self.num_class = 3
        elif self.dataset == "FND" and fold == 0: # two class setting
            self.num_class = 2
        elif self.dataset == "FND" and fold == 1: # four class setting
            self.num_class = 4
        elif self.dataset == "FC":
            self.num_class = 2
        elif self.dataset == "RCVP":
            self.num_class = 2
        else:
            print("nope")
            exit() 

        if self.lm_name == "roberta" or self.lm_name == "deberta" or self.lm_name == "bart":
            self.text_dim = 768
        elif self.lm_name == "electra":
            self.text_dim = 256

        self.input_process_layer = model.InputProcess(self.lm_name, self.dataset, self.in_channels)
        self.RingLM_seq = nn.Sequential()
        for i in range(hyperparams["ringlm_layer"]):
            self.RingLM_seq.append(model.RingLM(self.in_channels, hyperparams["nhead"], hyperparams["dropout_p"], self.dataset))
        if self.dataset == "SemEval" or self.dataset == "Allsides" or self.dataset == "FND":
            self.LinearOut = nn.Linear(3 * self.in_channels, self.num_class)
        elif self.dataset == "FC" or self.dataset == "RCVP":
            self.LinearOut = nn.Linear(2 * self.in_channels, self.num_class)
        self.CELoss = nn.CrossEntropyLoss()
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(hyperparams["dropout_p"])
        
        self.valmaxacc = 0
        self.valmaxf1 = 0

        if self.dataset == "RCVP":
            self.linear_rcvp = nn.Sequential(nn.Linear(512, self.in_channels), self.activation, self.dropout, nn.Linear(self.in_channels, self.in_channels), self.activation, self.dropout)
            self.RingLM2in = nn.Sequential(nn.Linear(3 * self.in_channels, self.in_channels), self.activation, self.dropout, nn.Linear(self.in_channels, self.in_channels), self.activation, self.dropout)

        if self.dataset == "FC":
            self.weight_text = nn.Sequential(nn.Linear(self.text_dim, self.in_channels), self.activation, self.dropout)
            self.weight_knowledge = nn.Sequential(nn.Linear(self.text_dim, self.in_channels), self.activation, self.dropout)
            self.weight_graph = nn.Sequential(nn.Linear(self.text_dim, self.in_channels), self.activation, self.dropout)
            self.multiheadatt_text = nn.MultiheadAttention(self.in_channels, hyperparams["nhead"])
            self.multiheadatt_knowledge = nn.MultiheadAttention(self.in_channels, hyperparams["nhead"])
            self.multiheadatt_graph = nn.MultiheadAttention(self.in_channels, hyperparams["nhead"])
            self.pre_out = nn.Sequential(nn.Linear(3 * self.in_channels, self.in_channels), self.activation, self.dropout)
            self.summary_out = nn.Sequential(nn.Linear(self.text_dim, self.in_channels), self.activation, self.dropout)


    def forward(self, input):
        x = self.input_process_layer(input)
        x = self.RingLM_seq(x)

        if self.dataset == "FC":
            summary_vec = self.input_process_layer.lm_extract(input["summary"]) # 1 * 768
            text_rep, _ = self.multiheadatt_text(self.weight_text(summary_vec), x["text"], x["text"]) # 1 * 512
            knowledge_rep, _ = self.multiheadatt_knowledge(self.weight_knowledge(summary_vec), x["knowledge"]["node_features"], x["knowledge"]["node_features"]) # 1 * 512
            graph_rep, _ = self.multiheadatt_graph(self.weight_graph(summary_vec), x["graph"]["node_features"], x["graph"]["node_features"]) # 1 * 512
            x = self.pre_out(torch.cat((torch.cat((text_rep, knowledge_rep), dim = 1), graph_rep), dim = 1))
            y = self.summary_out(summary_vec)
            x = torch.cat((y, x), dim = 1)
        else:
            x = torch.cat((torch.cat((torch.mean(x["text"],dim=0).unsqueeze(0), torch.mean(x["knowledge"]["node_features"],dim=0).unsqueeze(0)), dim = 1), torch.mean(x["graph"]["node_features"],dim=0).unsqueeze(0)), dim=1)
        
        
        if self.dataset == "RCVP": # concat(summary, doc body processed)
            summary_vec = self.linear_rcvp(input["leg_rep"]) # k * 768
            x = torch.cat((summary_vec, self.RingLM2in(x).repeat(len(summary_vec), 1)), dim = 1)
        
        x = self.LinearOut(x)
        return x # logit form

    def configure_optimizers(self):
        thing = [{"params": self.input_process_layer.model.parameters(), "lr": self.lr_lm},
                 {"params": self.input_process_layer.LinearT.parameters()}, {"params": self.input_process_layer.LinearK.parameters()},
                 {"params": self.input_process_layer.LinearG.parameters()}, {"params": self.RingLM_seq.parameters()},
                 {"params": self.LinearOut.parameters()}]
        if self.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(thing, lr = self.lr, weight_decay = self.weight_decay)
        elif self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(thing, lr = self.lr, weight_decay = self.weight_decay)
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(thing, lr = self.lr, weight_decay = self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 10)
        return ([optimizer], [scheduler])

    def on_epoch_start(self):
        if self.current_epoch == 0:
            #self.model.freeze()
            for param in self.input_process_layer.model.parameters():
                param.requires_grad = False
        
        if self.current_epoch == self.lm_freeze_epochs:
            #self.model.unfreeze()
            for param in self.input_process_layer.model.parameters():
                param.requires_grad = True
    
    def training_step(self, train_batch, batch_idx):
        batch_loss = 0
        truth = []
        pred = []
        for input in train_batch:
            if self.dataset == "RCVP":
                truth += input["label"]
                logit = self.forward(input)
                pred += [int(x) for x in list(torch.argmax(logit, dim=1))]
                loss = self.CELoss(logit, torch.tensor(input["label"]).long().cuda())
                batch_loss += loss
            else:
                truth.append(input["label"])
                logit = self.forward(input)
                pred.append(int(torch.argmax(logit, dim=1)))
                loss = self.CELoss(logit, torch.tensor([input["label"]]).long().cuda())
                batch_loss += loss
        acc = accuracy_score(truth, pred)
        if self.num_class == 2:
            f1 = f1_score(truth, pred)
        else:
            f1 = f1_score(truth, pred, average = "macro")
        batch_loss /= len(train_batch)
        return batch_loss

    def validation_step(self, val_batch, batch_idx):
        batch_loss = 0
        truth = []
        pred = []
        for input in val_batch:
            if self.dataset == "RCVP":
                truth += input["label"]
                logit = self.forward(input)
                pred += [int(x) for x in list(torch.argmax(logit, dim=1))]
                loss = self.CELoss(logit, torch.tensor(input["label"]).long().cuda())
                batch_loss += loss
            else:
                truth.append(input["label"])
                logit = self.forward(input)
                pred.append(int(torch.argmax(logit, dim=1)))
                loss = self.CELoss(logit, torch.tensor([input["label"]]).long().cuda())
                batch_loss += loss
        if self.dataset == "FC":
            acc = balanced_accuracy_score(truth, pred)
        else:
            acc = accuracy_score(truth, pred)
        if self.num_class == 2:
            f1 = f1_score(truth, pred)
        else:
            f1 = f1_score(truth, pred, average = "macro")
        if self.dataset == "RCVP":
            f1 = f1_score(truth, pred, average = "macro")
        batch_loss /= len(val_batch)
        #print("val acc: " + str(acc) + " val f1: " + str(f1))
        self.log("val_loss", batch_loss)
        self.log("val_acc", acc)
        if acc > self.valmaxacc:
            self.valmaxacc = acc
            self.valmaxf1 = f1
        #     if self.dataset == "RCVP" and acc > 0.85:
        #         f = open("logs/log_main_RCVP_verbose.txt", "a")
        #         f.write("Fold: " + str(self.fold) + " Acc: " + str(acc) + " MaF: " + str(f1) + "\n")
        #         f.close()
        return batch_loss

    def test_step(self, test_batch, batch_idx):
        batch_loss = 0
        truth = []
        pred = []
        for input in test_batch:
            if self.dataset == "RCVP":
                truth += input["label"]
                logit = self.forward(input)
                pred += [int(x) for x in list(torch.argmax(logit, dim=1))]
                loss = self.CELoss(logit, torch.tensor(input["label"]).long().cuda())
                batch_loss += loss
            else:
                truth.append(input["label"])
                logit = self.forward(input)
                pred.append(int(torch.argmax(logit, dim=1)))
                loss = self.CELoss(logit, torch.tensor([input["label"]]).long().cuda())
                batch_loss += loss
        if self.dataset == "FC":
            acc = balanced_accuracy_score(truth, pred)
        else:
            acc = accuracy_score(truth, pred)
        if self.num_class == 2:
            f1 = f1_score(truth, pred, average = "binary")
        else:
            f1 = f1_score(truth, pred, average = "macro")
        if self.dataset == "FC":
            f1 = f1_score(truth, pred, average = "micro")
        if self.dataset == "RCVP":
            f1 = f1_score(truth, pred, average = "macro")
        mif1 = f1_score(truth, pred, average = "micro")
        mapre = precision_score(truth, pred, average = "macro")
        marec = recall_score(truth, pred, average = "macro")

        # if self.num_class == 2:
        #     f1 = f1_score(truth, pred)
        # else:
        #     f1 = f1_score(truth, pred, average = "macro")
        batch_loss /= len(test_batch)
        #self.log("test_acc", acc)

        # logging
        f = open(self.log_dir, "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        if self.dataset == "SemEval" or self.dataset == "Allsides" or self.dataset == "FC" or self.dataset == "RCVP":
            f.write("Dataset: " + self.dataset + " Fold: " + str(self.fold) + " Time: " + dt_string + "\n")
            f.write("Val accuracy: " + str(self.valmaxacc) + " Val F1-score: " + str(self.valmaxf1) + "\n")
            f.write("Test accuracy: " + str(acc) + " Test F1-score: " + str(f1) + "\n")
            f.write("--------------------\n")
            f.close()
        elif self.dataset == "FND":
            f.write("Dataset: " + self.dataset + " Fold: " + str(self.fold) + " Time: " + dt_string + "\n")
            f.write("Val accuracy: " + str(self.valmaxacc) + " Val F1-score: " + str(self.valmaxf1) + "\n")
            f.write("Test mif1: " + str(mif1) + " Test maf1: " + str(f1) + " Test mapre: " + str(mapre) + " Test marec: " + str(marec) + "\n")
            f.write("--------------------\n")
            f.close()


def train_once(dataset, fold, hyperparams):
    train_loader, dev_loader, test_loader = dataloader.get_dataloaders(dataset, fold, hyperparams["batch_size"])
    model = RingLM_trainer(dataset, fold, hyperparams)
    #if dataset == "RCVP":
    #early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=hyperparams["patience"], verbose=True, mode="max")
    #else:
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=hyperparams["patience"], verbose=True, mode="min")
    if dataset == "SemEval":
        trainer = pl.Trainer(gpus = hyperparams["gpus"], num_nodes = 1, precision=16, max_epochs = hyperparams["max_epochs"], callbacks=[early_stop_callback], gradient_clip_val=1)
    elif dataset == "Allsides":
        trainer = pl.Trainer(gpus = hyperparams["gpus"], num_nodes = 1, precision=16, max_epochs = hyperparams["max_epochs"], callbacks=[early_stop_callback], gradient_clip_val=1)
    elif dataset == "FND":
        trainer = pl.Trainer(gpus = hyperparams["gpus"], num_nodes = 1, precision=16, max_epochs = hyperparams["max_epochs"], callbacks=[early_stop_callback], val_check_interval = 0.25, gradient_clip_val=1)
    elif dataset == "FC":
        trainer = pl.Trainer(gpus = hyperparams["gpus"], num_nodes = 1, precision=16, max_epochs = hyperparams["max_epochs"], callbacks=[early_stop_callback], gradient_clip_val=1)
    elif dataset == "RCVP":
        trainer = pl.Trainer(gpus = hyperparams["gpus"], num_nodes = 1, precision=16, max_epochs = hyperparams["max_epochs"], callbacks=[early_stop_callback], gradient_clip_val=1, auto_lr_find=True, num_sanity_val_steps=0)
    trainer.fit(model, train_loader, dev_loader)
    trainer.test(model, test_loader)