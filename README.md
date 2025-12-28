## Get Started

This is the source code of KBS paper *Structure adversarial augmented graph anomaly detection via multi-view 
contrastive learning* (SAA-GCL). The datasets are included in the Data/ folder.

## Code structure
| Folder |             Description             |
|:------:|:-----------------------------------:|
|  Data  |              Datasets.              |
| SAAGCL | The implementation code of SAA-GCL. |

## Datasets

|   **Dataset**   | # Nodes | # Edges | # Attribute | # Anomalies |
|:---------------:|:-------:|:-------:|:-----------:|:-----------:|
|    **Cora**     |  2,708  |  5,429  |    1,433    |    5.5%     |
|  **Citeseer**   |  3,327  |  4,732  |    3,703    |    4.5%     |
|   **Pubmed**    | 19,717  | 44,338  |     500     |    3.0%     |
|     **ACM**     | 16,484  | 71,980  |    8,337    |    3.6%     |
| **BlogCatalog** |  5,196  | 171,743 |    8,189    |    5.8%     |
|   **Reddit**    | 10,984  | 168,016 |     64      |    3.3%     |


## Usage
```
python ./SAAGCL/run.py --dataset 'reddit' --lr 4e-3 --train_epoch 100 --threshold 8 --alpha 0.2 --tau 0.2 --gpu_id 0
```


## Requirements
This code requires the following:

- python>=3.8
- pyTorch>=1.12.1
- scipy>=1.10.1
- tqdm>=4.67.1
- numpy>=1.24.3
- networkx>=2.7
- dgl<=0.4.3.post1

## Cite