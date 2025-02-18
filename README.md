# RNA-FM
**Update March 2024**: 

1. [Tutorials for RNA family clustering and RNA type classification](/tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb) & [Tutorial video (in Chinese)](https://www.bilibili.com/video/BV11D4215795/?vd_source=a80c1513b9533b969f95a485ab252511).
2. mRNA-FM, a foundation model pre-trained on coding sequences (CDS) in mRNA is now released! The model can take into CDSs and represent them with contextual embeddings, benefiting mRNA and protein related tasks.

This repository contains codes and pre-trained models for **RNA foundation model (RNA-FM)**.
**RNA-FM outperforms all tested single-sequence RNA language models across a variety of structure prediction tasks as well as several function-related tasks.**
You can find more details about **RNA-FM** in our paper, ["Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions" (Chen et al., 2022).](https://arxiv.org/abs/2204.00300)

![Overview](./docs/pics/overview.png)


<details><summary>Citation</summary>

```bibtex
@article{chen2022interpretable,
  title={Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}
```
</details>

<details><summary>Table of contents</summary>
  
- [Setup Environment](#Setup_Environment)
- [Pre-trained Models](#Available_Pretrained_Models)
- [Usage](#usage)
  - [RNA-FM Embedding Generation](#RNA-FM_Embedding_Generation)
  - [RNA Secondary Structure Prediction](#RNA_Secondary_Structure_Prediction)
  - [Server](#Server)
  - [Quick Start](#Quick_Start)
- [Related RNA Language Models](#Review)
- [Citations](#citations)
- [License](#license)
</details>

## Create Environment with Conda <a name="Setup_Environment"></a>
First, download the repository and create the environment.
```
git clone https://github.com/ml4bio/RNA-FM.git
cd ./RNA-FM
conda env create -f environment.yml
```
Then, activate the "RNA-FM" environment and enter into the workspace.
```
conda activate RNA-FM
cd ./redevelop
```
## Access pre-trained models. <a name="Available_Pretrained_Models"></a>
Download pre-trained models from [this gdrive link](https://drive.google.com/drive/folders/1VGye74GnNXbUMKx6QYYectZrY7G2pQ_J?usp=share_link) and place the pth files into the `pretrained` folder.

## Apply RNA-FM with Existing Scripts. <a name="Usage"></a>
### 1. Embedding Extraction. <a name="RNA-FM_Embedding_Generation"></a>
```
python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1 --save_embeddings
```
RNA-FM embeddings with shape of (L,640) will be saved in the `$save_dir/representations`.

As For mRNA-FM, you can call it with an extra argument, `MODEL.BACKBONE_NAME`:
```
python launch/predict.py --config="pretrained/extract_embedding.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1 --save_embeddings --save_embeddings_format raw MODEL.BACKBONE_NAME mrna-fm
```

### 2. Downstream Prediction - RNA secondary structure. <a name="RNA_Secondary_Structure_Prediction"></a>
```
python launch/predict.py --config="pretrained/ss_prediction.yml" \
--data_path="./data/examples/example.fasta" --save_dir="./resuts" \
--save_frequency 1
```
The predicted probability maps will be saved in form of `.npy` files, and the post-processed binary predictions will be saved in form of `.ct` files. You can find them in the `$save_dir/r-ss`.

### 3. Online Version - RNA-FM server. <a name="Server"></a>
If you have any trouble with the deployment of the local version of RNA-FM, you can access its online version from this link, [RNA-FM server](https://proj.cse.cuhk.edu.hk/rnafm/#/).
You can easily submit jobs on the server and download results from it afterwards, without setting up environment and occupying any computational resources.


## Quick Start for Further Development. <a name="Quick_Start"></a>
Python 3.8 (maybe higher version) and PyTorch are the prerequisite packages which you must have installed to use this repository.
You can install `rna-fm` in your own environment with the following pip command if you just want to
use the pre-trained language model. 
you can either install rna-fm from PIPY:
```
pip install rna-fm
```
or install `rna-fm` from github:
```
cd ./RNA-FM
pip install .
```
After installation, you can load the RNA-FM and extract its embeddings with the following code:
```
import torch
import fm

# Load RNA-FM model
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data
data = [
    ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
    ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract embeddings (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
token_embeddings = results["representations"][12]
```
More tutorials can be found from [https://ml4bio.github.io/RNA-FM/](https://ml4bio.github.io/RNA-FM/). The related notebooks are stored in the `tutorials` folder. 

As for mRNA-FM, the above code needs a slight revision. To be noted, the length of input RNA sequences should be the multiple of 3 to ensure the sequence can be tokenized into a series of codons (3-mer).
```
import torch
import fm

# Load mRNA-FM model
model, alphabet = fm.pretrained.mrna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data
data = [
    ("CDS1", "AUGGGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCUA"),
    ("CDS2", "AUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ("CDS3", "AUGCGAUUCNCGUUCCC--CCGCCUCC"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Extract embeddings (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
token_embeddings = results["representations"][12]
```

## Related RNA Language Models (BERT-style) <a name="Review"></a>
| Shorthand | Code | Subject | Layers | Embed Dim | Max Length | Input | Token | Dataset | Description | Year | Publisher |
|-----------|------|-------|--------|------|-----|-----|-------|-------| ---- | -------| ---- | 
| [RNA-FM](https://doi.org/10.48550/arXiv.2204.00300) | [Yes](https://github.com/ml4bio/RNA-FM) | ncRNA | 12 | 640 | 1024 | Seq |base | RNAcentral 19 (23 million samples) | The first RNA language model for general purpose | 2022.04 | arxiv/bioRxiv |
| [RNABERT](https://doi.org/10.1093/nargab/lqac012) | [Yes](https://github.com/mana438/RNABERT)   | ncRNA  | 6 | 120 | 440 | Seq | base | RNAcentral (762370) & Rfam 14.3 dataset (trained with partial MSA）| Specialized in structural alignment and clustering | 2022.02  | NAR Genomics and Bioinformatics |
| [UNI-RNA](https://doi.org/10.1101/2023.07.11.548588) | No  | RNA     | 24 | 1280 | $\infty$ |  Seq   | base | RNAcentral & nt & GWH (1 billion) | A general model with larger scale and datasets than RNA-FM | 2023.07 |  bioRxiv |
| [RNA-MSM](https://doi.org/10.1093/nar/gkad1031)| [Yes](https://github.com/yikunpku/RNA-MSM)   | ncRNA   | 12 | 768 | 1024 |  MSA   | base | 4069 RNA families from Rfam 14.7 | A model utilize evolutionary information from MSA directly | 2023.03 | NAR |
| [SpliceBERT](https://doi.org/10.1101/2023.01.31.526427) | [Yes](https://github.com/biomedAI/SpliceBERT) | pre-mRNA | 6 | 1024 | 512 | Seq  | base | 2 million precursor messenger RNA (pre-mRNA) | Specialized in RNA splicing of pre-mRNA | 2023.05 | bioRxiv |
| [CodonBERT]((https://doi.org/10.1101/2023.09.09.556981)) | No | mRNA CDS | 12 | 768 | 512*2 | Seq  | codon (3mer) | 10 million mRNAs from NCBI | Only focus on CDS of mRNA without UTRs | 2023.09 | bioRxiv |
| [UTR-LM](https://doi.org/10.1101/2023.10.11.561938) | [Yes](https://github.com/a96123155/UTR-LM)  | mRNA 5'UTR | 6 | 128 | $\infty$ | Seq | base | 700K 5'UTRs from Ensembl & eGFP & mCherry & Cao | Used for 5'UTR and mRNA expression related tasks | 2023.10 | bioRxiv |
| [3UTRBERT](https://doi.org/10.1101/2023.09.08.556883) | [Yes](https://github.com/yangyn533/3UTRBERT)  | mRNA 3'UTR | 12 | 768 | 512 | Seq | k-mer | 20,362 3'UTRs | Used for 3'UTR mediated gene regulation tasks |  2023.09 | bioRxiv |
| [BigRNA](https://doi.org/10.1101/2023.09.20.558508) | No  | DNA | - | - | - | Seq | - | thousands of genome-matched datasets | tissue-specific RNA expression, splicing, microRNA sites, and RNA binding protein |  2023.09 | bioRxiv |

## Citations <a name="citations"></a>

If you find our models useful in your research, we ask that you cite the relevant papers:

For **RNA Structure Prediction**:

```bibtex
@article{shen2024accurate,
  title={Accurate RNA 3D structure prediction using a language model-based deep learning approach},
  author={Shen, Tao and Hu, Zhihang and Sun, Siqi and Liu, Di and Wong, Felix and Wang, Jiuming and Chen, Jiayang and Wang, Yixuan and Hong, Liang and Xiao, Jin and others},
  journal={Nature Methods},
  year={2024}
}
```

```bibtex
@article{chen2020rna,
  title={RNA Secondary Structure Prediction By Learning Unrolled Algorithms},
  author={Chen, X. and Li, Y. and Umarov, R. and Gao, X. and Song, L.},
  journal={Proceedings of the Eighth International Conference on Learning Representations (ICLR)},
  year={2020}
}
```
For **RNA Design and Inverse Folding**:
```bibtex
@article{wong2024deep,
  title={Deep generative design of RNA aptamers using structural predictions},
  author={Wong, F. and He, D. and Krishnan, A. and Hong, L. and Wang, J. and Hu, Z. and others},
  journal={Nature Computational Science},
  year={2024}
}
```

```bibtex
@article{huang2024ribodiffusion,
  title={RiboDiffusion: Tertiary Structure-based RNA Inverse Folding with Generative Diffusion Models},
  author={Huang, H. and Lin, Z. and He, D. and Hong, L. and Li, Y.},
  journal={Bioinformatics},
  year={2024}
}
```
For **RNA-Protein Interaction (RPI) Prediction**:

```bibtex
@article{wei2023rna,
  title={RNA-Protein Interaction Prediction Based on Deep Learning},
  author={Wei, J. and Xiao, J. and Chen, S. and Zong, L. and Gao, X. and Li, Y.},
  journal={Journal Name},
  year={2023}
}
```

```bibtex
@article{wei2022structure,
  title={Protein-RNA interaction prediction with deep learning: Structure matters},
  author={Wei, J. and Chen, S. and Zong, L. and Gao, X. and Li, Y.},
  journal={Briefing in Bioinformatics},
  year={2022}
}
```

```bibtex
@article{lam2019deep,
  title={A deep learning framework to predict binding preference of RNA constituents on protein surface},
  author={Lam, J. et al.},
  journal={Nature Communications},
  year={2019}
}
```
For **Databases and Resources**:
```bibtex
@article{wei2024pronetdb,
  title={ProNet DB: A proteome-wise database for protein surface property representations and RNA-binding profiles},
  author={Wei, J. et al.},
  journal={Database},
  year={2024}
}
```
For **Single-Cell RNA Sequencing Analysis**:
```bibtex
@article{han2022self,
  title={Self-supervised contrastive learning for integrative single cell RNA-seq data analysis},
  author={Han, W., Cheng, Y., Chen, J., Zhong, H., Hu, Z., Chen, S., Zong, L., Hong, L., Chan, T.F., King, I., Gao, X., Li,Y.},
  journal={Briefing in Bioinformatics},
  year={2022}
}
```

For **Drug Discovery**:
```bibtex
@article{fan2022conserved,
  title={The highly conserved RNA-binding specificity of nucleocapsid protein facilitates the identification of drugs with broad anti-coronavirus activity},
  author={Fan,S., Sun,W., Fan,L., Wu,N., Ma,H., Chen,S., Li,Z., Li,Y., Zhang,J., Yan,J.},
  journal={Computational and Structural Biotechnology Journal},
  year={2022}
}
```
We appreciate your support in citing our work!

The model of this code builds on the [esm](https://github.com/facebookresearch/esm) sequence modeling framework. 
And we use [fairseq](https://github.com/pytorch/fairseq) sequence modeling framework to train our RNA language modeling.
We very appreciate these two excellent works!

## License <a name="license"></a>

This source code is licensed under the MIT license found in the `LICENSE` file
in the root directory of this source tree.
