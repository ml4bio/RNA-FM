# RNA-FM: The RNA Foundation Model
[![Pic](./docs/pics/RNA-FM.png)](https://proj.cse.cuhk.edu.hk/rnafm/#/)

[![arXiv](https://img.shields.io/badge/arXiv-2204.00300-b31b1b.svg)](https://arxiv.org/abs/2204.00300)
[![Nature Methods](https://img.shields.io/badge/Nature_Methods-10.1038/s41592--024--02487--0-1f77b4.svg)](https://www.nature.com/articles/s41592-024-02487-0)
[![Nature Computational Science](https://img.shields.io/badge/Nature_Computational_Science-10.1038/s43588--024--00720--6-1f77b4.svg)](https://www.nature.com/articles/s43588-024-00720-6)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-10.1093/bioinformatics/btab616-0887f7.svg)](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903)
[![RNA-FM Server](https://img.shields.io/badge/RNA_FM%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
[![RhoFold Server](https://img.shields.io/badge/RhoFold%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/#/)

## Introduction

[**RNA-FM** (RNA Foundation Model)](https://arxiv.org/abs/2204.00300) is a state-of-the-art **pretrained language model for RNA sequences**, serving as the cornerstone of an integrated RNA research ecosystem. Trained on **23+ million non-coding RNA (ncRNA) sequences** via self-supervised learning, RNA-FM extracts comprehensive structural and functional information from RNA sequences *without* relying on experimental labels. Consequently, RNA-FM generates **general-purpose RNA embeddings** suitable for a broad range of downstream tasks, including but not limited to secondary and tertiary structure prediction, RNA family clustering, and functional RNA analysis.

Originally introduced in [*Nature Methods*](https://arxiv.org/abs/2204.00300) as a foundational model for RNA biology, RNA-FM outperforms all evaluated single-sequence RNA language models across a wide reange of structure and function benchmarks, enabling unprecedented accuracy in RNA analysis. Building upon this foundation, our team developed an **integrated RNA pipeline** that includes:

- [**RhoFold**](https://www.nature.com/articles/s41592-024-02487-0) – High-accuracy RNA tertiary structure prediction (sequence → structure).  
- [**RiboDiffusion**](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903) – Diffusion-based inverse folding for RNA 3D design (structure → sequence).  
- [**RhoDesign**](https://www.nature.com/articles/s43588-024-00720-6) – Geometric deep learning approach to RNA design (structure → sequence).

These tools work alongside RNA-FM to **predict RNA structures from sequence, design new RNA sequences which could fold into desired 3D structures, and analyze functional properties**. Our integrated ecosystem is built to **advance the development of RNA therapeutics, drive innovation in synthetic biology, and deepen our understandings of RNA structure-function relationships**.

<details><summary>References</summary>

```bibtex
@article{chen2022interpretable,
  title={Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}
```
</details>

<details open><summary><b>Table of Contents</b></summary>

- [Introduction](#introduction)
- [RNA-FM and Related Tools](#rna-fm-and-related-tools)
  - [RNA-FM (Foundation Model)](#rna-fm-foundation-model)
  - [Downstream Tools](#downstream-tools)
    - [RhoFold (Tertiary Structure Prediction)](#rhofold-tertiary-structure-prediction)
    - [RiboDiffusion (Inverse Folding – Diffusion)](#ribodiffusion-inverse-folding--diffusion)
    - [RhoDesign (Inverse Folding – Deterministic)](#rhodesign-inverse-folding--deterministic)
- [Applications](#applications)
- [Setup and Usage](#setup-and-usage)
  - [Setup Environment with Conda](#setup-environment-with-conda)
  - [Quick Start Usage](#quick-start-usage)
  - [Online Server](#online-server)
- [Further Development & Python API](#further-development--python-api)
  - [Tutorials](#tutorials)
  - [Usage Examples with the Ecosystem](#usage-examples-with-the-ecosystem)
  - [API Reference](#api-reference)
- [Related RNA Language Models](#related-rna-language-models)
- [Citations](#citations)
- [License](#license)

</details>

---
## RNA-FM and Related Tools

**RNA-FM Ecosystem Components**: Our platform comprises four integrated tools, each addressing a critical step in the RNA analysis and design pipeline:

| Model | Task                                                                                                                     | Description                                                                                                          | Code | Paper                                                                                         |
|-------|--------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------|-----------------------------------------------------------------------------------------------|
| **RNA-FM** | **Foundation Model** (Representation)                                                                                    | Pretrained transformer (BERT-style) for ncRNA sequences; extracts embeddings and predicts base-pairing probabilities | [GitHub](https://github.com/ml4bio/RNA-FM) | [Nature Methods](https://arxiv.org/abs/2204.00300)|
| **RhoFold** | 3D Structure Prediction | RNA-FM-powered model for sequence-to-structure prediction (3D coordinates + secondary structure)                     | [GitHub](https://github.com/ml4bio/RhoFold) | [Nature Methods](https://www.nature.com/articles/s41592-024-02487-0)|
| **RiboDiffusion** | Inverse Folding                                                                                                          | Generative diffusion model for structure-to-sequence RNA design                                                      | [GitHub](https://github.com/ml4bio/RiboDiffusion) | [ISMB'2024](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903) |
| **RhoDesign** | Inverse Folding                                                                                                          | Geometric deep learning model (GVP+Transformer) for structure-to-sequence design                                     | [GitHub](https://github.com/ml4bio/RhoDesign) | [Nature Computational Science](https://www.nature.com/articles/s43588-024-00720-6)|

### RNA-FM (Foundation Model)
- [**RNA-FM (Foundation Model)**](https://github.com/ml4bio/RNA-FM) is a BERT-style Transformer model (12 layers, 640 hidden dimensions) trained on millions of RNA sequences. RNA-FM learns general-purpose RNA representations that encode both structural and functional information. It provides APIs for embedding extraction and can directly predict base-pairing probabilities for RNA secondary structure.

  <details open><summary>Click to fold RNA-FM details</summary>
  <br>
  
  [![CUHKServer](https://img.shields.io/badge/CUHK%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
  [![arXiv](https://img.shields.io/badge/arXiv-2204.00300-b31b1b.svg)](https://arxiv.org/abs/2204.00300)
    
    RNA-FM is pre-trained on massive RNA sequence data (RNAcentral) to produce contextual embeddings. These embeddings fuel structure-related tasks (e.g., secondary structure prediction, 3D distance/closeness prediction) and function-related tasks (e.g., UTR function, RNA-protein interaction). The RNA-FM model (12-layer Transformer) is at the core of both pre-training and fine-tuning stages, providing generalizable representations. Downstream, specialized tools (RhoFold, RiboDiffusion, RhoDesign) leverage RNA-FM for end-to-end RNA engineering.
  
  [![RNA-FM Overview](./docs/pics/overview.png)](https://github.com/ml4bio/RNA-FM)

  - **RNA-FM** for Secondary Structure Prediction:
    - Outperforms classic physics-based and machine learning methods (e.g., LinearFold, SPOT-RNA, UFold) by up to **20–30%** in F1-score on challenging datasets.
    - Performance gains are especially notable for long RNAs (>150 nucleotides) and low-homology families
    
  </details>

### Downstream Tools

#### RhoFold (Tertiary Structure Prediction)

- [**RhoFold (Tertiary Structure Prediction)**](https://github.com/ml4bio/RhoFold) – An RNA-FM–powered predictor for RNA 3D structures. Given an RNA sequence, RhoFold rapidly predicts its tertiary structure (3D coordinates in PDB format) along with the secondary structure (CT file) and per-residue confidence scores. It achieves high accuracy on RNA 3D benchmarks by combining RNA-FM embeddings with a structure prediction network, significantly outperforming prior methods in the RNA-Puzzles challenge.

  <details><summary>Click to expand RhoFold details</summary>
   <br>

    [![CUHKServer](https://img.shields.io/badge/CUHK%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/)
    [![Nature Methods](https://img.shields.io/badge/Nature_Methods-10.1038/s41592--024--02487--0-1f77b4.svg)](https://www.nature.com/articles/s41592-024-02487-0)

    RhoFold leverages the powerful embeddings from RNA-FM to revolutionize RNA tertiary structure prediction. By combining deep learning with structural biology principles, RhoFold translates RNA sequences directly into accurate 3D coordinates. The model employs a multi-stage architecture that first converts RNA-FM's contextual representations into distance maps and torsion angles, then assembles these into complete three-dimensional structures. Unlike previous approaches that often struggle with RNA's complex folding landscapes, RhoFold's foundation model approach captures subtle sequence-structure relationships, enabling state-of-the-art performance on challenging benchmarks like RNA-Puzzles. The system works in both single-sequence mode for rapid predictions and can incorporate multiple sequence alignments (MSA) when higher accuracy is needed, making it versatile for various research applications from small RNAs to complex ribozymes and riboswitches.

  [![RhoFlod Overview](https://github.com/ml4bio/RhoFold/raw/main/View.png)](https://github.com/ml4bio/RhoFold)
  - **RhoFold** for Tertiary Structure:
    - Delivers top accuracy on RNA-Puzzles / CASP-type tasks.
    - Predicts 3D structures **within seconds** (single-sequence mode) and integrates MSA for further accuracy gains.
    - Achieved *Nature Methods*–level benchmarks, generalizing to novel RNA families.
  </details>

#### RiboDiffusion (Inverse Folding – Diffusion)

- [**RiboDiffusion (Inverse Folding – Diffusion)**](https://github.com/ml4bio/RiboDiffusion) – A diffusion-based inverse folding model for RNA design. Starting from a target 3D backbone structure, RiboDiffusion iteratively generates RNA sequences that fold into that shape. This generative approach yields higher sequence recovery (≈11–16% improvement) than previous inverse folding algorithms, while offering tunable diversity in the designed sequences.

  <details><summary>Click to expand RiboDiffusion details</summary>
  <br>

  [![Bioinformatics](https://img.shields.io/badge/Bioinformatics-10.1093/bioinformatics/btab616-0887f7.svg)](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903)

    RiboDiffusion represents a breakthrough in RNA inverse folding through diffusion-based generative modeling. While traditional RNA design methods often struggle with the vast sequence space, RiboDiffusion employs a novel approach inspired by recent advances in generative AI. Starting with random noise, the model iteratively refines RNA sequences to conform to target 3D backbones through a carefully controlled diffusion process. This approach allows RiboDiffusion to explore diverse sequence solutions while maintaining structural fidelity, a critical balance in biomolecular design. The diffusion framework inherently provides sequence diversity, enabling researchers to generate and test multiple candidate designs that all satisfy structural constraints. Published benchmarks demonstrate that RiboDiffusion achieves superior sequence recovery rates compared to previous methods, making it particularly valuable for designing functional RNAs like riboswitches, aptamers, and other structured elements where sequence-structure relationships are crucial.

  [![Overview](https://github.com/ml4bio/RiboDiffusion/raw/main/fig/pipeline.png)](https://github.com/ml4bio/RiboDiffusion)

  - **RiboDiffusion** for Inverse Folding:
    - A diffusion-based generative approach that surpasses prior methods by **~11–16%** in sequence recovery rate.
    - Provides **tunable diversity** in design, exploring multiple valid sequences for a single target shape.

  </details>

#### RhoDesign (Inverse Folding – Deterministic)

- [**RhoDesign (Inverse Folding – Deterministic)**](https://github.com/ml4bio/RhoDesign) – A deterministic geometric deep learning model for RNA design. RhoDesign uses graph neural networks (GVP) and Transformers to directly decode sequences for a given 3D structure (optionally incorporating secondary structure constraints). It achieves state-of-the-art accuracy in matching target structures, with sequence recovery rates exceeding 50% on standard benchmarks (nearly double traditional methods) and the highest structural fidelity (TM-scores) among current solutions.

  <details><summary>Click to expand RhoDesign details</summary>
  <br>

    [![Nature Computational Science](https://img.shields.io/badge/Nature_Computational_Science-10.1038/s43588--024--00720--6-1f77b4.svg)](https://www.nature.com/articles/s43588-024-00720-6)

  RhoDesign introduces a deterministic approach to RNA inverse folding using geometric deep learning. Unlike diffusion-based methods, RhoDesign directly translates 3D structural information into RNA sequences through a specialized architecture combining Graph Vector Perceptrons (GVP) and Transformer networks. This architecture effectively captures both local geometric constraints and global structural patterns in RNA backbones. RhoDesign can incorporate optional secondary structure constraints, allowing researchers to specify certain base-pairing patterns while letting the model optimize the remaining sequence. Benchmark tests demonstrate that RhoDesign achieves remarkable sequence recovery rates exceeding 50% on standard datasets—nearly double the performance of traditional methods. Moreover, the designed sequences exhibit the highest structural fidelity (as measured by TM-score) among current approaches. This combination of accuracy and efficiency makes RhoDesign particularly suitable for precision RNA engineering applications where structural integrity is paramount.

  [![Overview](https://github.com/ml4bio/RhoDesign/raw/main/model_arc.png)](https://github.com/ml4bio/RhoDesign)

  - **RhoDesign** for Inverse Folding:
    - A deterministic GVP + Transformer model with **>50%** sequence recovery on standard 3D design benchmarks, nearly double that of older algorithms.
    - Achieves highest structural fidelity (TM-score) among tested methods, validated in *Nature Computational Science*.

  </details>

**Unified Workflow**: These tools operate in concert to enable end-to-end RNA engineering. For any RNA sequence of interest, one can **predict its structure** (secondary and tertiary) using RNA-FM and RhoFold. Conversely, given a desired RNA structure, one can **design candidate sequences** using RiboDiffusion or RhoDesign (or both for cross-validation). Designed sequences can then be validated by folding them back with RhoFold, closing the loop. This forward-and-inverse design cycle, all powered by RNA-FM embeddings, creates a powerful closed-loop workflow for exploring RNA structure-function space. By seamlessly integrating prediction and design, the RNA-FM ecosystem accelerates the design-build-test paradigm in RNA science, laying the groundwork for breakthroughs in RNA therapeutics, synthetic biology constructs, and our understanding of RNA biology.

---

## Applications

### RNA 3D Structure Prediction
- **Accurate RNA 3D structure prediction using a language-model–based deep learning approach** – introduces **RhoFold+**, which couples RNA-FM embeddings with a geometry module to reach SOTA accuracy on CASP/RNA-Puzzles benchmarks ([PAPER](https://doi.org/10.1038/s41592-024-02487-0), [CODE](https://github.com/ml4bio/RhoFold))
- **NuFold: end-to-end RNA tertiary-structure prediction** – integrates RNA-FM features into a U-former backbone, achieving accuracy competitive with state-of-the-art fold predictors ([PAPER](https://doi.org/10.1038/s41467-025-56261-7), [CODE](https://github.com/kiharalab/NuFold))
- **TorRNA – improved backbone-torsion prediction by leveraging large language models** – uses RNA-FM as sequence encoder and cuts median torsion-angle error by 2–16 % versus previous methods ([PAPER](https://chemrxiv.org/engage/chemrxiv/article-details/6658568c91aefa6ce1586b2d)) 

### RNA Design & Inverse Folding
- **Deep generative design of RNA aptamers using structural predictions** – employs **RhoDesign** to create Mango aptamer variants with >3-fold fluorescence gain (wet-lab verified) ([PAPER](https://doi.org/10.1038/s43588-024-00720-6), [CODE](https://github.com/ml4bio/RhoDesign))
- **RiboDiffusion: tertiary-structure-based RNA inverse folding with generative diffusion models** – diffusion sampler trained on RhoFold-generated data; boosts native-sequence recovery by 11 – 16 % over secondary-structure baselines ([PAPER](https://doi.org/10.1093/bioinformatics/btae259), [CODE](https://github.com/ml4bio/RiboDiffusion))  
- **gRNAde: geometric deep learning for 3-D RNA inverse design** – validates every design by forward-folding with RhoFold, achieving 56 % native-sequence recovery vs 45 % for Rosetta ([PAPER](https://arxiv.org/abs/2305.14749), [CODE](https://github.com/chaitjo/geometric-rna-design))  
- **RILLIE framework** – integrates a 1.6 B-parameter RNA LM with **RhoDesign** for in-silico directed evolution of Broccoli/Pepper aptamers ([CODE](https://github.com/GENTEL-lab/RILLIE)) 

### Functional Annotation & Subcellular Localisation
- **RNALoc-LM: RNA subcellular localisation prediction with a pre-trained RNA language model** – replaces one-hot inputs with RNA-FM embeddings, raising MCC by 4–8 % for lncRNA, circRNA and miRNA localisation ([PAPER](https://doi.org/10.1093/bioinformatics/btaf127), [CODE](https://github.com/CSUBioGroup/RNALoc-LM))  
- **PlantRNA-FM: an interpretable RNA foundation model for plant transcripts** – adapts the RNA-FM architecture to >25 M plant RNAs; discovers translation-related structural motifs and attains F1 = 0.97 on genic-region annotation ([PAPER](https://doi.org/10.1038/s42256-024-00946-z), [CODE](https://github.com/yangheng95/PlantRNA-FM))

### RNA–Protein Interaction
- **ZHMolGraph: network-guided deep learning for RNA–protein interaction prediction** – combines RNA-FM (for RNAs) and ProtTrans (for proteins) embeddings within a GNN, boosting AUROC by up to 28 % on unseen RNA–protein pairs ([PAPER](https://doi.org/10.1038/s42003-025-07694-9), [CODE](https://github.com/Zhaolab-GitHub/ZHMolGraph))  

> **Take-away:**  Across structure prediction, *de novo* sequence design, functional annotation and interaction modelling, the community is steadily adopting **RNA-FM** and its **RhoFold/RiboDiffusion/RhoDesign** toolkit as reliable building blocks—demonstrating the ecosystem’s versatility and real-world impact.

---

## Setup and Usage

### Setup Environment with Conda

Below, we outline the environment setup for **RNA-FM** and its extended pipeline (e.g., RhoFold) locally.  
*(If you prefer not to install locally, refer to the [Online Server](#Server) mentioned earlier.)*

1. **Clone the repository and create the Conda environment**:

```bash
git clone https://github.com/ml4bio/RNA-FM.git
cd RNA-FM
conda env create -f environment.yml
```

2. **Activate and enter the workspace**:

```bash
conda activate RNA-FM
cd ./redevelop
```

3. **Download pre-trained models** from our [Google Drive folder](https://drive.google.com/drive/folders/1VGye74GnNXbUMKx6QYYectZrY7G2pQ_J?usp=share_link) and place the `.pth` files into the `pretrained` folder.  

    > For **mRNA-FM**, ensure that your input RNA sequences have lengths multiple of 3 (codons) and place the specialized weights for *mRNA-FM* in the same `pretrained` folder.


### Quick Start Usage

Once the environment is ready and weights are downloaded, you can perform common tasks as follows:

#### 1. Embedding Generation


Use **RNA-FM** to extract nucleotide-level embeddings for input sequences:

```bash
python launch/predict.py \
    --config="pretrained/extract_embedding.yml" \
    --data_path="./data/examples/example.fasta" \
    --save_dir="./results" \
    --save_frequency 1 \
    --save_embeddings
```

This command processes sequences in `example.fasta` and saves 640-dimensional embeddings per nucleotide to `./results/representations/`.

- *Using mRNA-FM:* To use the mRNA-FM variant instead of the default ncRNA model, add the model name argument and ensure input sequences are codon-aligned:
  
  ```bash
  python launch/predict.py \
      --config="pretrained/extract_embedding.yml" \
      --data_path="./data/examples/example.fasta" \
      --save_dir="./results" \
      --save_frequency 1 \
      --save_embeddings \
      --save_embeddings_format raw \
      MODEL.BACKBONE_NAME mrna-fm
  ```
  As For mRNA-FM, you can call it with an extra argument, `MODEL.BACKBONE_NAME`.
  Remember **mRNA-FM** uses codon tokenization, so each sequence must have a length divisible by 3.

#### 2. RNA Secondary Structure Prediction

Predict an RNA secondary structure (base-pairing) from sequence using RNA-FM:

```bash
python launch/predict.py \
    --config="pretrained/ss_prediction.yml" \
    --data_path="./data/examples/example.fasta" \
    --save_dir="./results" \
    --save_frequency 1
```

RNA-FM will output base-pair probability matrices (`.npy`) and secondary structures (`.ct`) to `./results/r-ss`.

### Online Server

[![RNA-FM Server](https://img.shields.io/badge/RNA_FM%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
[![RhoFold Server](https://img.shields.io/badge/RhoFold%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/#/)

If you prefer **not to install** anything locally, you can use our **[RNA-FM server](https://proj.cse.cuhk.edu.hk/rnafm/#/)**. The server provides a simple web interface where you can:

- Submit an RNA sequence to get its predicted secondary structure and/or embeddings.
- Obtain results without needing local compute resources or setup.

(A separate **[RhoFold server](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/#/)** is also available for tertiary structure prediction of single RNA sequences.)


## Further Development & Python API

If you only want to **use the pretrained model** (rather than run all pipeline scripts), you can install `RNA-FM` directly:

```bash
pip install rna-fm
```

Alternatively, for the latest version from GitHub:

```bash
cd ./RNA-FM
pip install .
```

Then, load **RNA-FM** within your own Python project:

```python
import torch
import fm

# 1. Load RNA-FM model
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# 2. Prepare data
data = [
    ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
    ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ("RNA3", "CGAUUCNCGUUCCC--CCGCCUCCA"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 3. Extract embeddings (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
token_embeddings = results["representations"][12]
```

For **mRNA-FM**, load with `fm.pretrained.mrna_fm_t12()` and ensure input sequences are codon-aligned (as shown in the Quick Start above).

```python
import torch
import fm

# 1. Load mRNA-FM model
model, alphabet = fm.pretrained.mrna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# 2. Prepare data
data = [
    ("CDS1", "AUGGGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCUA"),
    ("CDS2", "AUGGGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
    ("CDS3", "AUGCGAUUCNCGUUCCC--CCGCCUCC"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# 3. Extract embeddings (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
token_embeddings = results["representations"][12]
```

More tutorials can be found from [GitHub](https://ml4bio.github.io/RNA-FM/). The related notebooks are stored in the [tutorials](./tutorials) folder.

### Tutorials

Get started with RNA-FM through our comprehensive tutorials:

| Tutorial                                                                                                                                                                                                                                                                                                                                                         | Description                                                                                                                                                                                                                                                | Format                                                                                              |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| [RNA Family Clustering & Type Classification](./tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb) <br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml4bio/RNA-FM/blob/main/tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb) | How to extract RNA-FM embeddings for clustering RNA families and classifying RNA types. This tutorial covers visualization of embeddings and training simple classifiers on top of them.                                                                   | [Jupyter Notebook](./tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb) |
| [RNA Secondary Structure Prediction](./tutorials/secondary-structure-prediction/Secondary-Structure-Prediction.py)                                                                                                                                                                                                                                               | How to use RNA-FM to predict RNA secondary structures, output base-pairing probability matrices, and visualize the predicted base-pairing (secondary structure).                                                                                           | [Python Script](./tutorials/secondary-structure-prediction/Secondary-Structure-Prediction.py)       |
| [UTR Function Prediction](./tutorials/utr-function-prediction/UTR-Function-Prediction.ipynb) <br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml4bio/RNA-FM/blob/main/tutorials/utr-function-prediction/UTR-Function-Prediction.ipynb)                                                 | How to leverage RNA-FM embeddings to predict functional properties of untranslated regions (5′ and 3′ UTRs) in mRNAs. This includes training a model to predict gene expression or protein translation metrics from UTR sequences.                         | [Jupyter Notebook](./tutorials/utr-function-prediction/UTR-Function-Prediction.ipynb)                                                                                  |
| [mRNA Expression Prediction](./tutorials/mRNA_expression/mrnafm-tutorial-code.ipynb)<br/>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ml4bio/RNA-FM/blob/main/tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb#scrollTo=71d77ca2)                          | How to use mRNA-FM variant to predict gene expression levels from mRNA sequences. This tutorial demonstrates loading the specialized mRNA model, extracting embeddings, and building a classifier to differentiate between high and low expression genes.  | [Jupyter Notebook](./tutorials/mRNA_expression/mrnafm-tutorial-code.ipynb) |

**Additional Resources:**

- [<img src="https://www.svgrepo.com/download/345504/bilibili.svg" alt="Bilibili" width="16" height="16"> Video Tutorial (Chinese)](https://www.bilibili.com/video/BV11D4215795/?vd_source=a80c1513b9533b969f95a485ab252511) - Step-by-step guide to using RNA-FM for various RNA analysis tasks

These tutorials cover the core applications of RNA-FM from basic embedding extraction to advanced functional predictions. Each provides hands-on examples you can run immediately in your browser or local environment.


### Usage Examples with the Ecosystem

We recommend exploring the advanced **RhoFold**, **RiboDiffusion**, and **RhoDesign** projects for tasks like 3D structure prediction or RNA design. Below are *brief* usage samples:

<details><summary>Click to expand RNA-FM Ecosystem details</summary>

#### RhoFold (Sequence → Structure)

```bash
# Example: Predict 3D structure for an RNA sequence in FASTA.
cd RhoFold
python inference.py \
    --input_fas ./example/input/5t5a.fasta \
    --output_dir ./example/output/5t5a/ \
    --ckpt ./pretrained/RhoFold_pretrained.pt
```

**Outputs**:  
- `unrelaxed_model.pdb` / `relaxed_1000_model.pdb` (3D coordinates)  
- `ss.ct` (secondary structure)  
- `results.npz` (distance/angle predictions + confidence scores)  
- `log.txt` (run logs, pLDDT, etc.)

#### RiboDiffusion (Structure → Sequence)

```bash
cd RiboDiffusion
CUDA_VISIBLE_DEVICES=0 python main.py \
    --PDB_file examples/R1107.pdb \
    --config.eval.n_samples 5
```

This will generate 5 candidate RNA sequences that fold into the structure provided in `R1107.pdb`. The output FASTA files will be saved under the `exp_inf/fasta/` directory.

#### RhoDesign (Structure → Sequence)

```bash
cd RhoDesign
python src/inference.py \
    --pdb ../example/2zh6_B.pdb \
    --ss ../example/2zh6_B.npy \
    --save ../example/
```

This produces a designed RNA sequence predicted to fold into the target 3D shape (PDB file `2zh6_B.pdb`, with an optional secondary structure constraint from `2zh6_B.npy`). The output sequence will be saved in the specified folder. You can adjust parameters like the sampling temperature to explore more diverse or high-fidelity designs.

</details>

### API Reference

### API Reference

Each project in the RNA-FM ecosystem comes with both command-line interfaces and Python modules:

- **RNA-FM:** Core module `fm` for embedding extraction and secondary structure prediction.
  - `fm.pretrained.rna_fm_t12()` – load the 12-layer ncRNA model
  - `fm.pretrained.mrna_fm_t12()` – load the 12-layer mRNA (codon) model
- **RhoFold:** Use the `RhoFoldModel` class or the `inference.py` script.
  - `inference.py` takes a FASTA sequence (and optionally an MSA) and outputs a 3D structure.
  - Add `--single_seq_pred True` to run without an MSA (single-sequence mode).
- **RiboDiffusion:** Use the `main.py` script or import the diffusion model classes.
  - `main.py` takes a PDB structure as input and outputs designed sequences.
  - Modify settings in `configs/` (e.g., `cond_scale`, `n_samples`) to tune the generation.
- **RhoDesign:** Use the `inference.py` script or import the design model module.
  - `inference.py` takes a PDB (and optional secondary structure/contact map) and outputs a designed sequence.
  - The GVP+Transformer architecture can incorporate partial structure constraints and supports advanced sampling strategies.

For further details, see each repo’s documentation or the notebooks in the [tutorials](./tutrorials) folder.

---

## Related RNA Language Models

| Name | Dataset | Modality | Tokenization | Architecture | Backbone | Pre‑training Task | Layers | Model Params | Data Size | Code                                                       | Weights | Data | License |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |------------------------------------------------------------| --- | --- | --- |
| **[RNA‑FM](https://doi.org/10.48550/arXiv.2204.00300)** | ncRNA | Sequence | Base | Enc‑only | Transformer | MLM | 12 | 100 M | 23 M | [GitHub](https://github.com/ml4bio/RNA-FM)                 | [HuggingFace](https://huggingface.co/ml4bio/RNA-FM) | RNAcentral  | MIT |
| [RNABERT](https://doi.org/10.1093/nargab/lqac012) | ncRNA | Sequence | Base | Enc‑only | Transformer | MLM / SAL | 6 | 0.5 M | 0.76 M | [GitHub](https://github.com/mana438/RNABERT)               | [Drive](https://drive.google.com/drive/folders/1j4DBx3W489MQTDjVT6hmsycCy0UON860) | Rfam 14.3  | MIT |
| [RNA‑MSM](https://doi.org/10.1093/nar/gkad1031) | ncRNA | Seq + MSA | Base | Enc‑only | MSA‑Transformer | MLM | 12 | 95 M | 3932 families | [GitHub](https://github.com/yikunpku/RNA-MSM)              | [Drive](https://pan.baidu.com/s/1VuoAoXjkR517SUOOxGGnNQ) | Rfam 14.7  | MIT |
| [AIDO.RNA](https://doi.org/10.1101/2024.11.28.625345) | ncRNA | Sequence | Base | Enc‑only | Transformer | MLM | 32 | 1.6 B | 42 M | [GitHub](https://github.com/genbio-ai/AIDO)                | [HuggingFace](https://huggingface.co/genbio-ai/AIDO.RNA-1.6B) | Public ncRNA mix | Apache‑2.0 |
| [ERNIE‑RNA](https://doi.org/10.1101/2024.03.17.585376) | ncRNA | Sequence | Base | Enc‑only | Transformer | MLM | 12 | 86 M | 20.4 M | [GitHub](https://github.com/Bruce-ywj/ERNIE-RNA)           | [GitHub](https://github.com/Bruce-ywj/ERNIE-RNA) | Rfam + RNAcentral | MIT |
| [GenerRNA](https://doi.org/10.1371/journal.pone.0310814) | ncRNA | Sequence | BPE | Dec‑only | Transformer | CLM | 24 | 350 M | 16.09 M | [GitHub](https://github.com/pfnet-research/GenerRNA)       | [HuggingFace](https://huggingface.co/pfnet/GenerRNA) | Public ncRNA mix | Apache‑2.0 |
| [RFamLlama](https://doi.org/10.1093/nar/gkaa1047) | ncRNA | Sequence | Base | Dec‑only | Llama | CLM | 6‑10 | 13‑88 M | 0.6 M | [HuggingFace](https://huggingface.co/jinyuan22/RFamLlama-large)     | [HuggingFace](https://huggingface.co/jinyuan22/RFamLlama-large)   | Rfam 14.10  | CC BY‑NC‑4.0 |
| [RNA‑km](https://doi.org/10.1101/2024.01.27.577533) | ncRNA | Sequence | Base | Enc‑only | Transformer | MLM | 12 | 152 M | 23 M | [GitHub](https://github.com/gongtiansu/RNA-km)             | [Drive](https://drive.google.com/drive/folders/1hC2iu9HJiUh6yIJ7qef9p0NjRbbFwZRU) | Rfam + RNAcentral | MIT |
| [RNAErnie](https://doi.org/10.1038/s42256-024-00836-4) | ncRNA | Sequence | Base | Enc‑only | Transformer | MLM | 12 | 105 M | 23 M | [GitHub](https://github.com/CatIIIIIIII/RNAErnie)          | [GitHub](https://github.com/CatIIIIIIII/RNAErnie) | Public ncRNA mix | Apache‑2.0 |
| [OPED](https://doi.org/10.1038/s42256-023-00739-w) | pegRNA | Sequence | k‑mer | Enc‑Dec | Transformer | Regression | n/a | n/a | 40 k | [GitHub](https://github.com/wenjiegroup/OPED)              | — | Public pegRNA eff. | MIT |
| [GARNET](https://doi.org/10.1101/2024.04.05.588317) | rRNA | Sequence | k‑mer | Dec‑only | Transformer | CLM | 18 | 19 M | 89 M tokens | [GitHub](https://github.com/Doudna-lab/GARNET_DL)          | [Release](https://github.com/Doudna-lab/GARNET_DL/releases) | Public rRNA | MIT |
| [IsoCLR](https://doi.org/10.48550/arXiv.2402.05943) | pre‑mRNA | Sequence | One‑hot | Enc‑only | CNN | Contrast Learning | 8 | 1‑10 M | 1 M | [GitHub](https://github.com/isoform/isoCLR)                | — | Ensembl / RefSeq | — |
| [SpliceBERT](https://doi.org/10.1101/2023.01.31.526427) | pre‑mRNA | Sequence | Base | Enc‑only | Transformer | MLM | 6 | 20 M | 2 M | [GitHub](https://github.com/chenkenbio/SpliceBERT)         | [Zenodo](https://zenodo.org/record/7740373) | UCSC/GENCODE | MIT |
| [Orthrus](https://doi.org/10.1101/2024.10.10.617658) | pre‑mRNA | Sequence | Base | Enc‑only | Mamba | Contrast Learning | 3‑6 | 1‑10 M | 49 M | [GitHub](https://github.com/bowang-lab/Orthrus)            | [HuggingFace](https://huggingface.co/antichronology/orthrus-4track) | Ortholog set  | Apache‑2.0 |
| [LoRNA](https://doi.org/10.1101/2024.08.26.609813) | pre‑mRNA | Sequence | Base | Dec‑only | StripedHyena | Contrast Learning | 16 | 6.5 M | 100 M | [GitHub](https://github.com/goodarzilab/lorna-sh)          | (announced) | SRA (long‑read) | MIT |
| [CodonBERT](https://doi.org/10.1101/2023.09.09.556981) | mRNA CDS | Sequence | Codon | Enc‑only | Transformer | MLM / HSP | 12 | 87 M | 10 M | [GitHub](https://github.com/Sanofi-Public/CodonBERT)       | [HuggingFace](https://huggingface.co/Sanofi/CodonBERT) | NCBI mRNA | Apache‑2.0 |
| [UTR‑LM](https://doi.org/10.1101/2023.10.11.561938) | 5′UTR | Sequence | Base | Enc‑only | Transformer | MLM / SSP / MFE | 6 | 1 M | 0.7 M | [GitHub](https://github.com/a96123155/UTR-LM)              | [GitHub](https://github.com/a96123155/UTR-LM) | Public 5′UTR set | MIT |
| [3UTRBERT](https://doi.org/10.1101/2023.09.08.556883) | 3′UTR | Sequence | k‑mer | Enc‑only | Transformer | MLM | 12 | 86 M | 20 k | [GitHub](https://github.com/yangyn533/3UTRBERT)            | [HuggingFace](https://huggingface.co/yangyn533/3UTRBERT) | Public 3′UTR | MIT |
| [G4mer](https://doi.org/10.1101/2024.10.01.616124) | mRNA | Sequence | k‑mer | Enc‑only | Transformer | MLM | 6 | — | — | —                                                          | — | — | — |
| [HELM](https://doi.org/10.48550/arXiv.2410.12459) | mRNA | Sequence | Codon | Multi | Multi | MLM + CLM | — | 50 M | 15.3 M | —                                                          | — | — | — |
| [RiNALMo](https://doi.org/10.48550/arXiv.2403.00043) | RNA | Sequence | Base | Enc‑only | Transformer | MLM | 33 | 135‑650 M | 36 M | [GitHub](https://github.com/rjpenic/RiNALMo)               | (request) | Public ncRNA | MIT |
| [UNI‑RNA](https://doi.org/10.1101/2023.07.11.548588) | RNA | Sequence | Base | Enc‑only | Transformer | MLM | 24 | 400 M | 500 M | —                                                          | — | — | — |
| [ATOM‑1](https://doi.org/10.1101/2023.12.13.571579) | RNA | Sequence | Base | Enc‑Dec | Transformer | — | — | — | — | —                                                          | — | — | — |
| [BiRNA‑BERT](https://doi.org/10.1101/2024.07.02.601703) | RNA | Sequence | Base + BPE | Enc‑only | Transformer | MLM | 12 | 117 M | 36 M | [GitHub](https://github.com/buetnlpbio/BiRNA-BERT)         | [HuggingFace](https://huggingface.co/buetnlpbio/BiRNA-BERT) | Public ncRNA | MIT |
| [ChaRNABERT](https://doi.org/10.48550/arXiv.2411.11808) | RNA | Sequence | GBST | Enc‑only | Transformer | MLM | 6‑33 | 8‑650 M | 62 M | —                                                          | (8 M demo) | Public ncRNA | — |
| [DGRNA](https://doi.org/10.1101/2024.10.31.621427) | RNA | Sequence | Base | Enc‑only | Mamba | MLM | 12 | 100 M | 100 M | —                                                          | — | — | — |
| [LAMAR](https://doi.org/10.1101/2024.10.12.617732) | RNA | Sequence | Base | Enc‑only | Transformer | MLM | 12 | 150 M | 15 M | [GitHub](https://github.com/zhw-e8/LAMAR)                  | (announced) | Public ncRNA | MIT |
| [OmniGenome](https://doi.org/10.48550/arXiv.2407.11242) | RNA | Sequence, Structure | Base | Enc‑only | Transformer | MLM / Seq2Str / Str2Seq | 16‑32 | 52‑186 M | 25 M | [GitHub](https://github.com/yangheng95/OmniGenBench)       | [HuggingFace](https://huggingface.co/yangheng95/OmniGenBench) | Public multi‑omics | Apache‑2.0 |
| [PlantRNA‑FM](https://doi.org/10.1038/s42256-024-00946-z) | RNA | Sequence, Structure | Base | Enc‑only | Transformer | MLM / SSP / CLS | 12 | 35 M | 25 M | [HuggingFace](https://huggingface.co/yangheng/PlantRNA-FM) | [HuggingFace](https://huggingface.co/yangheng/PlantRNA-FM) | Plant RNA set | CC BY‑NC‑4.0 |
| [MP‑RNA](https://doi.org/10.18653/v1/2024.findings-emnlp.304) | RNA | Sequence, Structure | Base | Enc‑only | Transformer | SSP / SNMR / MRLM | 12 | 52‑186 M | 25 M | [GitHub](https://github.com/yangheng95/OmniGenBench)       | (planned) | Public ncRNA mix | Apache‑2.0 |

---

## Citations

If you use RNA-FM or any components of this ecosystem in your research, please cite the relevant papers. Below is a collection of key publications (in BibTeX format) covering the foundation model and associated tools:

<details open><summary><b>BibTeX Citations</b></summary>

### RNA-FM & RNA Structure Predictions

```bibtex
@article{chen2022interpretable,
  title={Interpretable RNA foundation model from unannotated data for highly accurate RNA structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and Shen, Tao and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}

@article{shen2024accurate,
  title={Accurate RNA 3D structure prediction using a language model-based deep learning approach},
  author={Shen, Tao and Hu, Zhihang and Sun, Siqi and Liu, Di and Wong, Felix and Wang, Jiuming and Chen, Jiayang and Wang, Yixuan and Hong, Liang and Xiao, Jin and others},
  journal={Nature Methods},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group US New York}
}

@article{chen2020rna,
  title={RNA secondary structure prediction by learning unrolled algorithms},
  author={Chen, Xinshi and Li, Yu and Umarov, Ramzan and Gao, Xin and Song, Le},
  journal={arXiv preprint arXiv:2002.05810},
  year={2020}
}

@article{WANG2025102991,
   title = {Deep learning for RNA structure prediction},
   author = {Jiuming Wang and Yimin Fan and Liang Hong and Zhihang Hu and Yu Li},
   journal = {Current Opinion in Structural Biology},
   year = {2025},
   doi = {https://doi.org/10.1016/j.sbi.2025.102991},
   url = {https://www.sciencedirect.com/science/article/pii/S0959440X25000090},
}
```

### RNA Design & Inverse Folding

```bibtex
@article{wong2024deep,
  title={Deep generative design of RNA aptamers using structural predictions},
  author={Wong, Felix and He, Dongchen and Krishnan, Aarti and Hong, Liang and Wang, Alexander Z and Wang, Jiuming and Hu, Zhihang and Omori, Satotaka and Li, Alicia and Rao, Jiahua and others},
  journal={Nature Computational Science},
  pages={1--11},
  year={2024},
  publisher={Nature Publishing Group US New York}
}

@article{huang2024ribodiffusion,
  title={RiboDiffusion: tertiary structure-based RNA inverse folding with generative diffusion models},
  author={Huang, Han and Lin, Ziqian and He, Dongchen and Hong, Liang and Li, Yu},
  journal={Bioinformatics},
  volume={40},
  number={Supplement\_1},
  pages={i347--i356},
  year={2024},
  publisher={Oxford University Press}
}
```

### RNA-Protein Interaction (RPI)

```bibtex
@article{wei2022protein,
  title={Protein--RNA interaction prediction with deep learning: structure matters},
  author={Wei, Junkang and Chen, Siyuan and Zong, Licheng and Gao, Xin and Li, Yu},
  journal={Briefings in bioinformatics},
  volume={23},
  number={1},
  pages={bbab540},
  year={2022},
  publisher={Oxford University Press}
}

@article{lam2019deep,
  title={A deep learning framework to predict binding preference of RNA constituents on protein surface},
  author={Lam, Jordy Homing and Li, Yu and Zhu, Lizhe and Umarov, Ramzan and Jiang, Hanlun and H{\'e}liou, Am{\'e}lie and Sheong, Fu Kit and Liu, Tianyun and Long, Yongkang and Li, Yunfei and others},
  journal={Nature communications},
  volume={10},
  number={1},
  pages={4941},
  year={2019},
  publisher={Nature Publishing Group UK London}
}
```

### Databases & Resources

```bibtex
@article{wei2024pronet,
  title={ProNet DB: a proteome-wise database for protein surface property representations and RNA-binding profiles},
  author={Wei, Junkang and Xiao, Jin and Chen, Siyuan and Zong, Licheng and Gao, Xin and Li, Yu},
  journal={Database},
  volume={2024},
  pages={baae012},
  year={2024},
  publisher={Oxford University Press UK}
}
```

### Single-Cell RNA Analysis

```bibtex
@article{han2022self,
  title={Self-supervised contrastive learning for integrative single cell RNA-seq data analysis},
  author={Han, Wenkai and Cheng, Yuqi and Chen, Jiayang and Zhong, Huawen and Hu, Zhihang and Chen, Siyuan and Zong, Licheng and Hong, Liang and Chan, Ting-Fung and King, Irwin and others},
  journal={Briefings in Bioinformatics},
  volume={23},
  number={5},
  pages={bbac377},
  year={2022},
  publisher={Oxford University Press}
}
```

### Drug Discovery

```bibtex
@article{fan2022highly,
  title={The highly conserved RNA-binding specificity of nucleocapsid protein facilitates the identification of drugs with broad anti-coronavirus activity},
  author={Fan, Shaorong and Sun, Wenju and Fan, Ligang and Wu, Nan and Sun, Wei and Ma, Haiqian and Chen, Siyuan and Li, Zitong and Li, Yu and Zhang, Jilin and others},
  journal={Computational and Structural Biotechnology Journal},
  volume={20},
  pages={5040--5044},
  year={2022},
  publisher={Elsevier}
}
```
</details>

---

## License

This source code is licensed under the **MIT** license found in the [LICENSE](./LICENSE) file in the root directory of this source tree.  

Our framework and model training were inspired by:
- [**esm**](https://github.com/facebookresearch/esm) (Facebook’s protein language modeling framework)  
- [**fairseq**](https://github.com/pytorch/fairseq) (PyTorch sequence modeling framework)  

We thank the authors of these works for providing excellent foundations for RNA-FM.

---

**Thank you for using RNA-FM!**  
For issues or questions, open a GitHub [Issue](https://github.com/ml4bio/RNA-FM/issues) or consult the [documentation](https://ml4bio.github.io/RNA-FM/). We welcome contributions and collaboration from the community.

