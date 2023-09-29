import trainer
import random
import os

dataset = "RCVP" # change
#fold = 0
hyperparams = {
    "optimizer": "RAdam",
    "lm_name": "bart",
    "in_channels": 512,
    "lr_lm": 1e-5,
    "lr": 1e-4, # 1e-3 or 1e-4
    "weight_decay": 1e-5,
    "lm_freeze_epochs": 1000,
    "log_dir": "logs/log_main_RCVP.txt", # change
    "ringlm_layer": 2,
    "nhead": 8,
    "dropout_p": 0.5, # change
    "batch_size": 4, # change
    "max_epochs": 100, # 50 for SemEval, 25 for Allsides, 3 for FND, 10 for FC, 100 for RCVP
    "patience": 10,
    "gpus": 1 # change
}

while True:
    #fold = random.randint(0,9)
    #fold = random.randint(0,2)
    #fold = 1 # 0 for two-class FND, 1 for four-class FND
    #fold = 0 # 0 for all, 1 for cnndm, 2 for xsum
    fold = 0 # 0 for random, 1 for time-based
    trainer.train_once(dataset, fold, hyperparams)