# RNA-FM: The RNA Foundation Model for Integrated RNA Analysis and Design
[![Pic](./docs/pics/RNA-FM.png)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
**Update March 2024**  
1. **Tutorials**  
   - [Tutorial for RNA family clustering and RNA type classification](./tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb)  
   - [Tutorial video (in Chinese)](https://www.bilibili.com/video/BV11D4215795/?vd_source=a80c1513b9533b969f95a485ab252511)  
2. **mRNA-FM**  
   - A foundation model pre-trained on coding sequences (CDS) is now released!  
   - `mRNA-FM` can tokenize CDS (3-mer) and represent them with contextual embeddings, benefiting mRNA- and protein-related tasks.  

[//]: # (---)

## Introduction

**RNA-FM** (RNA Foundation Model) is a state-of-the-art **pretrained language model for RNA sequences**, serving as the cornerstone of an integrated RNA research ecosystem. Trained on **23+ million non-coding RNA (ncRNA) sequences** via self-supervised learning, RNA-FM captures rich structural and functional signals from RNA sequences *without* requiring experimental labels. Consequently, RNA-FM provides **general-purpose RNA embeddings** that can be applied to a broad range of downstream tasks—such as secondary and tertiary structure prediction, RNA family clustering, and functional RNA analysis.

Originally introduced in [*Nature Methods*](https://arxiv.org/abs/2204.00300) as a foundational model for RNA biology, RNA-FM outperforms all tested single-sequence RNA language models across diverse RNA structure and function benchmarks, enabling unprecedented accuracy in RNA analysis. Building upon this backbone, our team developed an **integrated RNA pipeline** that includes:

- [**RhoFold**](https://www.nature.com/articles/s41592-024-02487-0) – High-accuracy RNA tertiary structure prediction (*sequence → structure*).  
- [**RiboDiffusion**](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903) – A diffusion-based inverse folding model (*structure → sequence*) for 3D RNA design.  
- [**RhoDesign**](https://www.nature.com/articles/s43588-024-00720-6) – A geometric deep learning approach for *structure → sequence* design with strong accuracy.

These tools work in concert with RNA-FM to **predict RNA structures from sequence, design new RNA sequences for target 3D structures, and analyze functional properties**. Our integrated ecosystem is designed to **revolutionize RNA therapeutics design, synthetic biology, and the understanding of RNA structure-function relationships**.

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

<details open><summary><b>Table of contents</b></summary>

- [Introduction](#introduction)
- [RNA-FM and Related Tool](#RNA-FM-and-Related-Tool)
  - [RNA-FM (Foundation Model)](#rna-fm-foundation-model)
  - [RhoFold (Tertiary Structure Prediction)](#rhofold-tertiary-structure-prediction)
  - [RiboDiffusion (Inverse Folding)](#ribodiffusion-inverse-folding)
  - [RhoDesign (Inverse Folding)](#rhodesign-inverse-folding)
- [Applications](#applications)
- [Setup and Usage](#setup-and-usage)
  - [Setup Environment with Conda](#setup-environment-with-conda)
  - [Quick Start Usage](#quick-start-usage)
  - [Online Server](#online-server)
- [Further Development](#further-development)
  - [Python API](#python-api)
  - [Usage Examples with the Ecosystem](#usage-examples-with-the-ecosystem)
  - [API Reference](#api-reference)
- [Related RNA Language Models](#related-rna-language-models)
- [Citations](#citations)
- [License](#license)

</details>

---
## RNA-FM and Related Tool

**RNA-FM Ecosystem Components**: Our platform comprises four integrated tools, each addressing a critical step in the RNA analysis and design pipeline:

| Model | Task | Description | Code | Paper                                                                                         |
|-------|------|-------------|------|-----------------------------------------------------------------------------------------------|
| **RNA-FM** | Representation Learning | Specialized model for coding sequences using 3-mer tokenization | [GitHub](https://github.com/ml4bio/RNA-FM) | [Nature Methods](https://arxiv.org/abs/2204.00300)|
| **RhoFold** | 3D Structure Prediction | RNA-FM-powered model for sequence-to-structure prediction | [GitHub](https://github.com/ml4bio/RhoFold) | [Nature Methods](https://www.nature.com/articles/s41592-024-02487-0)|
| **RiboDiffusion** | Inverse Folding | Generative diffusion model for structure-to-sequence design | [GitHub](https://github.com/ml4bio/RiboDiffusion) | [Bioinformatics](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903) |
| **RhoDesign** | Inverse Folding | Geometric deep learning approach for RNA design | [GitHub](https://github.com/ml4bio/RhoDesign) | [Nature Computational Science](https://www.nature.com/articles/s43588-024-00720-6)|


- [**RNA-FM (Foundation Model)**](https://github.com/ml4bio/RNA-FM) – A BERT-style Transformer (12 layers, 640 hidden dim) trained on millions of RNA sequences. It learns general-purpose RNA embeddings that encode structural and functional information. RNA-FM provides APIs for embedding extraction and can directly predict base-pairing probabilities for secondary structure.

  <details><summary>Click to expand RNA-FM details</summary>

    RNA-FM is pre-trained on massive RNA sequence data (RNAcentral) to produce contextual embeddings. These embeddings fuel structure-related tasks (e.g., secondary structure prediction, 3D distance/closeness prediction) and function-related tasks (e.g., UTR function, RNA-protein interaction). The RNA-FM model (12-layer Transformer) is at the core of both pre-training and fine-tuning stages, providing generalizable representations. Downstream, specialized tools (RhoFold, RiboDiffusion, RhoDesign) leverage RNA-FM for end-to-end RNA engineering.
  
  [![RNA-FM Overview](./docs/pics/overview.png)](https://github.com/ml4bio/RNA-FM)

  - **RNA-FM** for Secondary Structure Prediction:
    - Outperforms classic physics-based and machine learning methods (e.g., LinearFold, SPOT-RNA, UFold) by up to **20–30%** in F1-score on challenging datasets.
    - Particularly robust on long RNAs (>150 nt) and low-homology families.
    
  </details>

- [**RhoFold (Tertiary Structure Prediction)**](https://github.com/ml4bio/RhoFold) – An RNA-FM–powered predictor for RNA 3D structures. Given an RNA sequence, RhoFold rapidly predicts its tertiary structure (3D coordinates in PDB format) along with the secondary structure (CT file) and per-residue confidence scores. It achieves high accuracy on RNA 3D benchmarks by combining RNA-FM embeddings with a structure prediction network, significantly outperforming prior methods in the RNA-Puzzles challenge.

  <details><summary>Click to expand RhoFold details</summary>

    RhoFold leverages the powerful embeddings from RNA-FM to revolutionize RNA tertiary structure prediction. By combining deep learning with structural biology principles, RhoFold translates RNA sequences directly into accurate 3D coordinates. The model employs a multi-stage architecture that first converts RNA-FM's contextual representations into distance maps and torsion angles, then assembles these into complete three-dimensional structures. Unlike previous approaches that often struggle with RNA's complex folding landscapes, RhoFold's foundation model approach captures subtle sequence-structure relationships, enabling state-of-the-art performance on challenging benchmarks like RNA-Puzzles. The system works in both single-sequence mode for rapid predictions and can incorporate multiple sequence alignments (MSA) when higher accuracy is needed, making it versatile for various research applications from small RNAs to complex ribozymes and riboswitches.

  [![RhoFlod Overview](https://github.com/ml4bio/RhoFold/raw/main/View.png)](https://github.com/ml4bio/RhoFold)

  - **RhoFold** for Tertiary Structure:
    - Delivers top accuracy on RNA-Puzzles / CASP-type tasks.
    - Predicts 3D structures **within seconds** (single-sequence mode) and integrates MSA for further accuracy gains.
    - Achieved *Nature Methods*–level benchmarks, generalizing to novel RNA families.

  </details>

- [**RiboDiffusion (Inverse Folding – Diffusion)**](https://github.com/ml4bio/RiboDiffusion) – A diffusion-based inverse folding model for RNA design. Starting from a target 3D backbone structure, RiboDiffusion iteratively generates RNA sequences that fold into that shape. This generative approach yields higher sequence recovery (≈11–16% improvement) than previous inverse folding algorithms, while offering tunable diversity in the designed sequences.

  <details><summary>Click to expand RiboDiffusion details</summary>

    RiboDiffusion represents a breakthrough in RNA inverse folding through diffusion-based generative modeling. While traditional RNA design methods often struggle with the vast sequence space, RiboDiffusion employs a novel approach inspired by recent advances in generative AI. Starting with random noise, the model iteratively refines RNA sequences to conform to target 3D backbones through a carefully controlled diffusion process. This approach allows RiboDiffusion to explore diverse sequence solutions while maintaining structural fidelity, a critical balance in biomolecular design. The diffusion framework inherently provides sequence diversity, enabling researchers to generate and test multiple candidate designs that all satisfy structural constraints. Published benchmarks demonstrate that RiboDiffusion achieves superior sequence recovery rates compared to previous methods, making it particularly valuable for designing functional RNAs like riboswitches, aptamers, and other structured elements where sequence-structure relationships are crucial.

  [![Overview](https://github.com/ml4bio/RiboDiffusion/raw/main/fig/pipeline.png)](https://github.com/ml4bio/RiboDiffusion)

  - **RiboDiffusion** for Inverse Folding:
    - A diffusion-based generative approach that surpasses prior methods by **~11–16%** in sequence recovery rate.
    - Provides **tunable diversity** in design, exploring multiple valid sequences for a single target shape.

  </details>

- [**RhoDesign (Inverse Folding – Deterministic)**](https://github.com/ml4bio/RhoDesign) – A deterministic geometric deep learning model for RNA design. RhoDesign uses graph neural networks (GVP) and Transformers to directly decode sequences for a given 3D structure (optionally incorporating secondary structure constraints). It achieves state-of-the-art accuracy in matching target structures, with sequence recovery rates exceeding 50% on standard benchmarks (nearly double traditional methods) and the highest structural fidelity (TM-scores) among current solutions.

  <details><summary>Click to expand RhoDesign details</summary>

  RhoDesign introduces a deterministic approach to RNA inverse folding using geometric deep learning. Unlike diffusion-based methods, RhoDesign directly translates 3D structural information into RNA sequences through a specialized architecture combining Graph Vector Perceptrons (GVP) and Transformer networks. This architecture effectively captures both local geometric constraints and global structural patterns in RNA backbones. RhoDesign can incorporate optional secondary structure constraints, allowing researchers to specify certain base-pairing patterns while letting the model optimize the remaining sequence. Benchmark tests demonstrate that RhoDesign achieves remarkable sequence recovery rates exceeding 50% on standard datasets—nearly double the performance of traditional methods. Moreover, the designed sequences exhibit the highest structural fidelity (as measured by TM-score) among current approaches. This combination of accuracy and efficiency makes RhoDesign particularly suitable for precision RNA engineering applications where structural integrity is paramount.

  [![Overview](https://github.com/ml4bio/RhoDesign/raw/main/model_arc.png)](https://github.com/ml4bio/RhoDesign)

  - **RhoDesign** for Inverse Folding:
    - A deterministic GVP + Transformer model with **>50%** sequence recovery on standard 3D design benchmarks, nearly double that of older algorithms.
    - Achieves highest structural fidelity (TM-score) among tested methods, validated in *Nature Computational Science*.

  </details>

**Unified Workflow**: These tools operate in concert to enable end-to-end RNA engineering. For any RNA sequence of interest, one can **predict its structure** (secondary and tertiary) using RNA-FM and RhoFold. Conversely, given a desired RNA structure, one can **design candidate sequences** using RiboDiffusion or RhoDesign (or both for cross-validation). Designed sequences can then be validated by folding them back with RhoFold, closing the loop. This forward-and-inverse design cycle, all powered by RNA-FM embeddings, creates a powerful closed-loop workflow for exploring RNA structure-function space. By seamlessly integrating prediction and design, the RNA-FM ecosystem accelerates the design-build-test paradigm in RNA science, laying the groundwork for breakthroughs in RNA therapeutics, synthetic biology constructs, and our understanding of RNA biology.

---

## Applications

Our RNA-FM ecosystem has broad applications in:

- **RNA Therapeutics & Drug Design**  
  - Accelerate development of mRNA vaccines, siRNAs, ribozymes, aptamers.  
  - Predict structural stability, design new functional motifs *in silico*, and reduce trial-and-error cycles.

- **Synthetic Biology**  
  - Rationally engineer riboswitches, RNA sensors, or logic circuits.  
  - Generate candidate sequences with **RiboDiffusion** or **RhoDesign**, then confirm 3D structures using **RhoFold**.

- **Functional RNA Analysis & Genomics**  
  - Leverage RNA-FM embeddings for RNA family clustering, classification, and annotation.  
  - Explore viral RNA architectures (e.g., SARS-CoV-2) and potential drug-binding sites.

- **Education & Exploratory Research**  
  - Easily access advanced RNA AI through servers and notebooks.  
  - Visualize embeddings, predict structures, or design candidate RNAs in minutes.

With the rapid rise of RNA-based technologies (e.g., mRNA vaccines), these **scientific and commercial** avenues are expanding quickly. Our integrated pipeline aims to streamline the design-build-test cycle for RNA, fostering innovation in both academic and industrial settings.

---

## Setup Environment with Conda

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

   - For **mRNA-FM**, ensure that your input RNA sequences have lengths multiple of 3 (codons) and place the specialized weights for *mRNA-FM* in the same `pretrained` folder.

---

## Quick Start Usage


### 1. Embedding Generation


Use **RNA-FM** to extract embeddings for downstream tasks:

```bash
python launch/predict.py \
    --config="pretrained/extract_embedding.yml" \
    --data_path="./data/examples/example.fasta" \
    --save_dir="./results" \
    --save_frequency 1 \
    --save_embeddings
```

This command processes sequences in `example.fasta` and saves 640-dimensional embeddings per nucleotide to `./results/representations/`.

- To run **mRNA-FM** (instead of default RNA-FM):
  
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

Remember **mRNA-FM** uses codon tokenization, so each sequence must have a length multiple of 3.

### 2. RNA Secondary Structure Prediction

```bash
python launch/predict.py \
    --config="pretrained/ss_prediction.yml" \
    --data_path="./data/examples/example.fasta" \
    --save_dir="./results" \
    --save_frequency 1
```

RNA-FM will output base-pair probability matrices (`.npy`) and secondary structures (`.ct`) to `./results/r-ss`.

### 3. Online Server <a name="Server"></a>

For those who prefer **not** to install locally, we offer an [RNA-FM server](https://proj.cse.cuhk.edu.hk/rnafm/#/) that:
- Lets you submit RNA sequences in a web interface.
- Returns secondary structure predictions and/or embeddings.
- Eliminates the need for local environment setup or computational resources.

---

## Further Development & Python API

If you only want to **use the pretrained model** (rather than run all pipeline scripts), you can install `rna-fm` directly:

```bash
pip install rna-fm
```

Or install from GitHub:

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
model.eval()  # no dropout

# 2. Prepare data
data = [
    ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU"),
    ("RNA2", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
]
_, _, tokens = batch_converter(data)

# 3. Extract embeddings
with torch.no_grad():
    out = model(tokens, repr_layers=[12])
embeddings = out["representations"][12]  # shape: (batch=1, length, 640)
print(embeddings.shape)
```

For **mRNA-FM**, replace `rna_fm_t12()` with `mrna_fm_t12()` and ensure codon-aligned input sequences.

---

## Usage Examples with the Ecosystem

We recommend exploring the advanced **RhoFold**, **RiboDiffusion**, and **RhoDesign** projects for tasks like 3D structure prediction or RNA design. Below are *brief* usage samples:

<details><summary>Click to expand RNA-FM Ecosystem details</summary>

### RhoFold (Sequence → Structure)

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

### RiboDiffusion (Structure → Sequence)

```bash
cd RiboDiffusion
CUDA_VISIBLE_DEVICES=0 python main.py \
    --PDB_file examples/R1107.pdb \
    --config.eval.n_samples 5
```

Generates 5 new sequences matching `R1107.pdb`. Output FASTA files are placed in `exp_inf/fasta/`.

### RhoDesign (Structure → Sequence)

```bash
cd RhoDesign
python src/inference.py \
    --pdb ../example/2zh6_B.pdb \
    --ss ../example/2zh6_B.npy \
    --save ../example/
```

Produces a designed sequence that folds into the target 3D shape, leveraging geometry-based encoding. Adjust temperature or other parameters for more diversity or fidelity.

</details>

---

## API Reference

Each project includes:
- **CLI scripts** (`predict.py`, `inference.py`, etc.).
- **Python modules** for integration (`fm` for RNA-FM, `RhoFoldModel` for RhoFold, etc.).
- **Configuration files** to customize training/inference.

**Common Patterns**:
- **RNA-FM**  
  - `fm.pretrained.rna_fm_t12()` → loads 12-layer ncRNA model  
  - `fm.pretrained.mrna_fm_t12()` → loads mRNA-specific variant  
- **RhoFold**  
  - `inference.py` takes FASTA + optional MSA file → outputs 3D structure.  
  - Use `--single_seq_pred True` if you don’t want/have MSA data.  
- **RiboDiffusion**  
  - `main.py` takes a PDB → outputs designed sequences.  
  - Tweak parameters in `configs/` (e.g., `cond_scale`, `n_samples`).  
- **RhoDesign**  
  - `inference.py` takes a PDB and (optionally) a 2D structure/contact map → outputs designed sequences.  
  - The GVP-based approach can incorporate partial constraints, advanced sampling, etc.

For further details, see each repo’s documentation or the notebooks in the `tutorials` folder.

---

## Related RNA Language Models

[//]: # (| Shorthand                                         | Code                                                       | Subject    | Layers | Embed Dim | Max Length | Input  | Token  | Dataset                                  | Description                                                   | Year    | Publisher               |)

[//]: # (|---------------------------------------------------|------------------------------------------------------------|------------|--------|----------|-----------|--------|--------|-------------------------------------------|---------------------------------------------------------------|---------|-------------------------|)

[//]: # (| **[RNA-FM]&#40;https://doi.org/10.48550/arXiv.2204.00300&#41;** | [Yes]&#40;https://github.com/ml4bio/RNA-FM&#41;                    | ncRNA      | 12     | 640      | 1024      | Seq    | base   | RNAcentral19 &#40;23M&#41;                        | The first large-scale, general-purpose RNA language model     | 2022.04 | arXiv/bioRxiv          |)

[//]: # (| [RNABERT]&#40;https://doi.org/10.1093/nargab/lqac012&#41; | [Yes]&#40;https://github.com/mana438/RNABERT&#41;                  | ncRNA      | 6      | 120      | 440       | Seq    | base   | RNAcentral &#40;762k&#41; & Rfam14.3 &#40;partial MSA&#41;| Specialized in structural alignment and clustering            | 2022.02 | NAR Genomics Bioinfo   |)

[//]: # (| [UNI-RNA]&#40;https://doi.org/10.1101/2023.07.11.548588&#41;    | No                                                         | RNA        | 24     | 1280     | ∞         | Seq    | base   | 1B from RNAcentral, nt, GWH               | Larger scale & dataset than RNA-FM; general model             | 2023.07 | bioRxiv                |)

[//]: # (| [RNA-MSM]&#40;https://doi.org/10.1093/nar/gkad1031&#41;   | [Yes]&#40;https://github.com/yikunpku/RNA-MSM&#41;                 | ncRNA      | 12     | 768      | 1024      | MSA    | base   | 4069 families from Rfam14.7               | Uses evolutionary information from MSA                        | 2023.03 | NAR                     |)

[//]: # (| [SpliceBERT]&#40;https://doi.org/10.1101/2023.01.31.526427&#41; | [Yes]&#40;https://github.com/biomedAI/SpliceBERT&#41;             | pre-mRNA   | 6      | 1024     | 512       | Seq    | base   | 2M pre-mRNAs                              | Specialized in RNA splicing                                  | 2023.05 | bioRxiv                |)

[//]: # (| [CodonBERT]&#40;https://doi.org/10.1101/2023.09.09.556981&#41;  | No                                                         | mRNA CDS   | 12     | 768      | 512×2     | Seq    | codon | 10M mRNAs from NCBI                       | Only for coding sequences; tokenized by codon                 | 2023.09 | bioRxiv                |)

[//]: # (| [UTR-LM]&#40;https://doi.org/10.1101/2023.10.11.561938&#41;     | [Yes]&#40;https://github.com/a96123155/UTR-LM&#41;                 | mRNA 5'UTR | 6      | 128      | ∞         | Seq    | base   | 700k 5'UTRs                               | Targets 5'UTR expression-related tasks                       | 2023.10 | bioRxiv                |)

[//]: # (| [3UTRBERT]&#40;https://doi.org/10.1101/2023.09.08.556883&#41;   | [Yes]&#40;https://github.com/yangyn533/3UTRBERT&#41;               | mRNA 3'UTR | 12     | 768      | 512       | Seq    | k-mer  | 20,362 3'UTRs                             | For 3'UTR-mediated gene regulation tasks                     | 2023.09 | bioRxiv                |)

[//]: # (| BigRNA &#40;WIP&#41;                                        | No                                                         | DNA→RNA    | -      | -        | -         | Seq    | -      | Thousands of genome-matched sets          | Tissue-specific expression, splicing, miRNA sites, RBP        | 2023.09 | bioRxiv                |)

[//]: # (---)

| Name | Dataset | Modality | Tokenization | Architecture | Backbone | Pre-training Task | Layers | Model Params | Data Size | Open Source                                        |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |----------------------------------------------------|
| **[RNA-FM](https://doi.org/10.48550/arXiv.2204.00300)** | ncRNA | Sequence | Base | Enc.only | Transformers | MLM | 12 | 100M | 23M | [Yes](https://github.com/ml4bio/RNA-FM)            |
| [RNABERT](https://doi.org/10.1093/nargab/lqac012) | ncRNA | Sequence | Base | Enc.only | Transformers | MLM, SAL | 6 | 0.5M | 0.76M | [Yes](https://github.com/mana438/RNABERT)          |
| [RNA-MSM](https://doi.org/10.1093/nar/gkad1031) | ncRNA | Sequence, MSA | Base | Enc.only | MSA Transformers | MLM | 12 | 95M | 3932 families | [Yes](https://github.com/yikunpku/RNA-MSM)         |
| [AIDO.RNA](https://doi.org/10.1101/2024.11.28.625345) | ncRNA | Sequence | Base | Enc.only | Transformers | MLM | 32 | 1.6B | 42M | [Yes](https://github.com/genbio-ai/AIDO)           |
| [ERNIE-RNA](https://doi.org/10.1101/2024.03.17.585376) | ncRNA | Sequence | Base | Enc.only | Transformers | MLM | 12 | 86M | 20.4M | [Yes](https://github.com/Bruce-ywj/ERNIE-RNA)      |
| [GenerRNA](https://doi.org/10.1371/journal.pone.0310814) | ncRNA | Sequence | BPE | Dec.only | Transformers | CLM | 24 | 350M | 16.09M | [Yes](https://github.com/pfnet-research/GenerRNA)  |
| [RFamLlama](https://doi.org/10.1093/nar/gkaa1047) | ncRNA | Sequence | Base | Dec.only | Llama | CLM | 6 - 10 | 13M - 88M | 0.6M | [Yes](https://github.com/Rfam)                     |
| [RNA-km](https://doi.org/10.1101/2024.01.27.577533) | ncRNA | Sequence | Base | Enc.only | Transformers | MLM | 12 | 152M | 23M | [Yes](https://github.com/gongtiansu/RNA-km)        |
| [RNAErnie](https://doi.org/10.1038/s42256-024-00836-4) | ncRNA | Sequence | Base | Enc.only | Transformers | MLM | 12 | 105M | 23M | [Yes](https://github.com/CatIIIIIIII/RNAErnie)     |
| [OPED](https://doi.org/10.1038/s42256-023-00739-w) | pegRNA | Sequence | k-mer | Enc.-Dec. | Transformers | Regression | N.A. | N.A. | 40k | [Yes](https://github.com/wenjiegroup/OPED)         |
| [GARNET](https://doi.org/10.1101/2024.04.05.588317) | rRNA | Sequence | k-mer | Dec.only | Transformers | CLM | 18 | 19M | 89M tokens | [Yes](https://github.com/Doudna-lab/GARNET_DL)     |
| [IsoCLR](https://doi.org/10.48550/arXiv.2402.05943) | pre-mRNA | Sequence | One-hot | Enc.only | CNN | CL | 8 | 1 - 10M | 1M | [Yes](unknown)                                     |
| [SpliceBERT](https://doi.org/10.1101/2023.01.31.526427) | pre-mRNA | Sequence | Base | Enc.only | Transformers | MLM | 6 | 20M | 2M | [Yes](https://github.com/chenkenbio/SpliceBERT)    |
| [Orthrus](https://doi.org/10.1101/2024.10.10.617658) | pre-mRNA | Sequence | Base | Enc.only | Mamba | CL | 3 - 6 | 1 - 10M | 49M | [Yes](https://github.com/bowang-lab/Orthrus)       |
| [LoRNA](https://doi.org/10.1101/2024.08.26.609813) | pre-mRNA | Sequence | Base | Dec.only | StripedHyena | CL | 16 | 6.5M | 100M | [Yes](https://github.com/goodarzilab/lorna-sh)     |
| [CodonBERT](https://doi.org/10.1101/2023.09.09.556981) | mRNA CDS | Sequence | Codon | Enc.only | Transformers | MLM, HSP | 12 | 87M | 10M | [Yes](https://github.com/Sanofi-Public/CodonBERT)  |
| [UTR-LM](https://doi.org/10.1101/2023.10.11.561938) | mRNA 5'UTR | Sequence | Base | Enc.only | Transformers | MLM, SSP, MFE | 6 | 1M | 0.7M | [Yes](https://github.com/a96123155/UTR-LM)         |
| [3UTRBERT](https://doi.org/10.1101/2023.09.08.556883) | mRNA 3'UTR | Sequence | k-mer | Enc.only | Transformers | MLM | 12 | 86M | 20k | [Yes](https://github.com/yangyn533/3UTRBERT)       |
| [G4mer](https://doi.org/10.1101/2024.10.01.616124) | mRNA | Sequence | k-mer | Enc.only | Transformers | MLM | 6 | N.A. | N.A. | No                                                 |
| [HELM](https://doi.org/10.48550/arXiv.2410.12459) | mRNA | Sequence | Codon | Multiple | Multiple | MLM, CLM | Multiple | 50M | 15.3M | No                                                 |
| [RiNALMo](https://doi.org/10.48550/arXiv.2403.00043) | RNA | Sequence | Base | Enc.only | Transformers | MLM | 33 | 135 - 650M | 36M | [Yes](https://github.com/rjpenic/RiNALMo)          |
| [UNI-RNA](https://doi.org/10.1101/2023.07.11.548588) | RNA | Sequence | Base | Enc.only | Transformers | MLM | 24 | 400M | 500M | No                                                 |
| [ATOM-1](https://doi.org/10.1101/2023.12.13.571579) | RNA | Sequence | Base | Enc.-Dec. | Transformers | N.A. | N.A. | N.A. | N.A. | No                                                 |
| [BiRNA-BERT](https://doi.org/10.1101/2024.07.02.601703) | RNA | Sequence | Base, BPE | Enc.only | Transformers | MLM | 12 | 117M | 36M | [Yes](https://github.com/buetnlpbio/BiRNA-BERT)    |
| [ChaRNABERT](https://doi.org/10.48550/arXiv.2411.11808) | RNA | Sequence | GBST | Enc.only | Transformers | MLM | 6 - 33 | 8 - 650M | 62M | [Yes](unknown)                                     |
| [DGRNA](https://doi.org/10.1101/2024.10.31.621427) | RNA | Sequence | Base | Enc.only | Mamba | MLM | 12 | 100M | 100M | No                                                 |
| [LAMAR](https://doi.org/10.1101/2024.10.12.617732) | RNA | Sequence | Base | Enc.only | Transformers | MLM | 12 | 150M | 15M | [Yes](https://github.com/zhw-e8/LAMAR)             |
| [OmniGenome](https://doi.org/10.48550/arXiv.2407.11242) | RNA | Sequence, Structure | Base | Enc.only | Transformers | MLM, Seq2Str, Str2Seq | 16 - 32 | 52M - 186M | 25M | [Yes](https://github.com/yangheng95/OmniGenBench)  |
| [PlantRNA-FM](https://doi.org/10.1038/s42256-024-00946-z) | RNA | Sequence, Structure | Base | Enc.only | Transformers | MLM, SSP, CLS | 12 | 35M | 25M | [Yes](https://huggingface.co/yangheng/PlantRNA-FM) |
| [MP-RNA](https://doi.org/10.18653/v1/2024.findings-emnlp.304) | RNA | Sequence, Structure | Base | Enc.only | Transformers | SSP, SNMR, MRLM | 12 | 52M - 186M | 25M | [Yes](https://github.com/yangheng95/OmniGenBench)  |

---

## Citations

If you use RNA-FM or any components of this ecosystem in your research, please cite the relevant papers. Below is a collection of key publications (in BibTeX format) covering the foundation model and associated tools:
<details><summary>Click to expand BibTeX citations</summary>

### RNA-FM & RNA Structure Predictions

```bibtex
@article{chen2022interpretable,
  title={Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}

@article{shen2024accurate,
  title={Accurate RNA 3D structure prediction using a language model-based deep learning approach},
  author={Shen, Tao and Hu, Zhihang and Sun, Siqi and Liu, Di and Wong, Felix and Wang, Jiuming and Chen, Jiayang and Wang, Yixuan and Hong, Liang and Xiao, Jin and others},
  journal={Nature Methods},
  year={2024}
}
```

### RNA Secondary Structure

```bibtex
@article{chen2020rna,
  title={RNA Secondary Structure Prediction By Learning Unrolled Algorithms},
  author={Chen, X. and Li, Y. and Umarov, R. and Gao, X. and Song, L.},
  journal={Proceedings of the Eighth International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

### RNA Design & Inverse Folding

```bibtex
@article{wong2024deep,
  title={Deep generative design of RNA aptamers using structural predictions},
  author={Wong, F. and He, D. and Krishnan, A. and Hong, L. and Wang, J. and Hu, Z. and others},
  journal={Nature Computational Science},
  year={2024}
}

@article{huang2024ribodiffusion,
  title={RiboDiffusion: Tertiary Structure-based RNA Inverse Folding with Generative Diffusion Models},
  author={Huang, H. and Lin, Z. and He, D. and Hong, L. and Li, Y.},
  journal={Bioinformatics},
  year={2024}
}
```

### RNA-Protein Interaction (RPI)

```bibtex
@article{wei2023rna,
  title={RNA-Protein Interaction Prediction Based on Deep Learning},
  author={Wei, J. and Xiao, J. and Chen, S. and Zong, L. and Gao, X. and Li, Y.},
  journal={Journal Name},
  year={2023}
}

@article{wei2022structure,
  title={Protein-RNA interaction prediction with deep learning: Structure matters},
  author={Wei, J. and Chen, S. and Zong, L. and Gao, X. and Li, Y.},
  journal={Briefing in Bioinformatics},
  year={2022}
}

@article{lam2019deep,
  title={A deep learning framework to predict binding preference of RNA constituents on protein surface},
  author={Lam, J. et al.},
  journal={Nature Communications},
  year={2019}
}
```

### Databases & Resources

```bibtex
@article{wei2024pronetdb,
  title={ProNet DB: A proteome-wise database for protein surface property representations and RNA-binding profiles},
  author={Wei, J. et al.},
  journal={Database},
  year={2024}
}
```

### Single-Cell RNA Analysis

```bibtex
@article{han2022self,
  title={Self-supervised contrastive learning for integrative single cell RNA-seq data analysis},
  author={Han, W. and Cheng, Y. and Chen, J. and Zhong, H. and Hu, Z. and Chen, S. and Zong, L. and Hong, L. and Chan, T.F. and King, I. and Gao, X. and Li,Y.},
  journal={Briefing in Bioinformatics},
  year={2022}
}
```

### Drug Discovery

```bibtex
@article{fan2022conserved,
  title={The highly conserved RNA-binding specificity of nucleocapsid protein facilitates the identification of drugs with broad anti-coronavirus activity},
  author={Fan,S., Sun,W., Fan,L., Wu,N., Ma,H., Chen,S., Li,Z., Li,Y., Zhang,J., Yan,J.},
  journal={Computational and Structural Biotechnology Journal},
  year={2022}
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

