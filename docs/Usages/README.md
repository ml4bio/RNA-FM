---
sort: 2
---
# Usages Principles

This section is used to give the principles for using our RNA-FM packages.

## 1. Embedding Extraction. <a name="RNA-FM_Embedding_Generation"></a>
```
python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1 --save_embeddings
```
RNA-FM embeddings with shape of (L,640) will be saved in the `$save_dir/representations`.

## 2. Downstream Prediction - RNA secondary structure. <a name="RNA_Secondary_Structure_Prediction"></a>
```
python launch/predict.py --config="pretrained/ss_prediction.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1
```
The predicted probability maps will be saved in form of `.npy` files, and the post-processed binary predictions will be saved in form of `.ct` files. You can find them in the `$save_dir/r-ss`.

## 3. Online Version - RNA-FM server. <a name="Server"></a>
If you have any trouble with the deployment of the local version of RNA-FM, you can access its online version from this link, [RNA-FM server](https://proj.cse.cuhk.edu.hk/rnafm/#/).
You can easily submit jobs on the server and download results from it afterwards, without setting up environment and occupying any computational resources.