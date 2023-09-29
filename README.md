## PoliLean Reposiotry

This is the official repo for [KALM: Knowledge-Aware Integration of Local, Document, and Global Contexts for Long Document Understanding](https://arxiv.org/abs/2210.04105) @ ACL 2023.

### Content

`process_graph.py`, `process_text.py`, and `process_knowledge.py` are data preprocessing codes for the three aspects.

`graph_gnn_layer.py` implements the knowledge-guided message passing GNN, `model.py` defines the modular components of KALM, `dataloader.py`, `trainer.py`, and `utils.py` are helper functions, `main.py` is the main executable file.

### Data link (raw and preprocessed)

preprocessed data: [link](https://drive.google.com/file/d/1Yy_blTM1UtgamBH-hgJqVix8jISeRpdQ/view?usp=sharing)

### Citation
If you find this repo useful, please cite our paper:
```
@inproceedings{feng-etal-2023-pretraining,
    title = "From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair {NLP} Models",
    author = "Feng, Shangbin  and
      Park, Chan Young  and
      Liu, Yuhan  and
      Tsvetkov, Yulia",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.656",
    doi = "10.18653/v1/2023.acl-long.656",
    pages = "11737--11762",
    abstract = "Language models (LMs) are pretrained on diverse data sources{---}news, discussion forums, books, online encyclopedias. A significant portion of this data includes facts and opinions which, on one hand, celebrate democracy and diversity of ideas, and on the other hand are inherently socially biased. Our work develops new methods to (1) measure media biases in LMs trained on such corpora, along social and economic axes, and (2) measure the fairness of downstream NLP models trained on top of politically biased LMs. We focus on hate speech and misinformation detection, aiming to empirically quantify the effects of political (social, economic) biases in pretraining data on the fairness of high-stakes social-oriented tasks. Our findings reveal that pretrained LMs do have political leanings which reinforce the polarization present in pretraining corpora, propagating social biases into hate speech predictions and media biases into misinformation detectors. We discuss the implications of our findings for NLP research and propose future directions to mitigate unfairness.",
}
```
