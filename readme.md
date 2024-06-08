# CauseLens: Causality-based Unsupervised Interpretable Root Cause Analysis for Microservice with Graph Autoencoder
## Abstract
```
Due to the complex invocation relationships among APIs in microservice applications, a single fault can propagate along multiple invocation paths, causing widespread failures. Different faults may have distinct propagation paths and manifestations, necessitating an efficient and interpretable Root Cause Analysis (RCA) method. Existing random walk-based methods may not be accurate enough, while many deep learning-based methods require sufficient labeled data, which is hard to obtain in a production environment. Moreover, both methods provide very limited interpretability.
We propose a causality-based unsupervised interpretable root cause analysis framework, CauseLens. The core idea is that modeling fine-grained causal relationships between microservices facilitates interpretable root cause localization. Thus, CauseLens constructs a heterogeneous causal diagram at the operation and entity levels using normal monitoring data (i.e., metrics and traces) of microservices and trains a structural causal model. It then employs a combination of reconstruction error and counterfactual analysis to pinpoint root causes while providing fault propagation paths.
Experimental results on two microservice datasets show that CauseLens outperforms state-of-the-art methods. We also conduct ablation studies and parameter experiments to confirm the design's effectiveness and test the method's overhead to ensure it meets real-time requirements.
```
## Overview

!['Overview'](.\\overview.png)
## **Files Architecture**

```
├── .vscode
│   └── launch.json			        # parameters setting
├── datasets                         # data directory (reserve to complete)
├── RCA
│   ├── images                              # experimental images
│   ├── logs                                # runtime logs and rca results                   
│   ├── model-store
│   │   └── TrainTicket 
│   │       └── CustomGAE
│   │            ├── rca_<date>	        # rca results
│   │            └── train_<date>	        # model file and intermediate files
│   │                ├── model.pth		     
│   │                └── run-info		     
│   ├── dataset_aiops22.py
│   ├── dataset.py
│   ├── dataset_trainticket.py
│   ├── layers.py
│   ├── log.py
│   ├── mask.py
│   ├── models.py
│   ├── rca.py
│   ├── run.py                      # entry file
│   └── utils.py
├── requirements.txt   
```
## **Run：**
1. Install requirements following requirements.txt.
2. Set parameters according to parameters settings in `.vscode`, run the entry file `run.py`.







