# RNA-FM: The RNA Foundation Model for Integrated RNA Analysis and Design

**Update March 2024**  
1. **Tutorials**  
   - [Tutorial for RNA family clustering and RNA type classification](./tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb)  
   - [Tutorial video (in Chinese)](https://www.bilibili.com/video/BV11D4215795/?vd_source=a80c1513b9533b969f95a485ab252511)  
2. **mRNA-FM**  
   - A foundation model pre-trained on coding sequences (CDS) is now released!  
   - `mRNA-FM` can tokenize CDS (3-mer) and represent them with contextual embeddings, benefiting mRNA- and protein-related tasks.  

---

## Introduction

**RNA-FM** (RNA Foundation Model) is a state-of-the-art **pretrained language model for RNA sequences**, serving as the cornerstone of an integrated RNA research ecosystem. Trained on **23+ million non-coding RNA (ncRNA) sequences** via self-supervised learning, RNA-FM captures rich structural and functional signals from RNA sequences *without* requiring experimental labels. Consequently, RNA-FM provides **general-purpose RNA embeddings** that can be applied to a broad range of downstream tasks—such as secondary and tertiary structure prediction, RNA family clustering, and functional RNA analysis.

Originally introduced in *Nature Methods* (Chen *et al.*, 2022) as a foundational model for RNA biology, RNA-FM outperforms all tested single-sequence RNA language models across diverse RNA structure and function benchmarks, enabling unprecedented accuracy in RNA analysis. Building upon this backbone, our team developed an **integrated RNA pipeline** that includes:

- **RhoFold** – High-accuracy RNA tertiary structure prediction (*sequence → structure*).  
- **RiboDiffusion** – A diffusion-based inverse folding model (*structure → sequence*) for 3D RNA design.  
- **RhoDesign** – A geometric deep learning approach for *structure → sequence* design with strong accuracy.

These tools work in concert with RNA-FM to **predict RNA structures from sequence, design new RNA sequences for target 3D structures, and analyze functional properties**. Our integrated ecosystem is designed to **revolutionize RNA therapeutics design, synthetic biology, and the understanding of RNA structure-function relationships**.

Below is a quick snapshot of how RNA-FM works and how it compares to other single-sequence models:

![Overview](./docs/pics/overview.png)

If you find our model or pipeline useful, please cite our paper,  
> *“Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions” (Chen et al., 2022).*  
> [arXiv:2204.00300](https://arxiv.org/abs/2204.00300)  

---

## Project Ecosystem

Beyond the standalone RNA-FM, we have built a tightly **integrated ecosystem** of cutting-edge RNA tools:

1. **RNA-FM (Foundation Model)**
   - BERT-style Transformer (12 layers, 640-dim) trained on millions of RNA sequences from RNAcentral.
   - Learns general-purpose RNA embeddings encoding structural and functional signals.
   - Provides easy-to-use APIs for embedding extraction and secondary structure predictions.

2. **RhoFold (3D Structure Prediction)**
   - RNA-FM–powered model for **sequence-to-structure** prediction.
   - Predicts RNA tertiary structures (PDB files), along with secondary structure `.ct` format and confidence scores.
   - Significantly outperforms existing methods on multiple RNA folding benchmarks.

3. **RiboDiffusion (Inverse Folding with Diffusion)**
   - Generative diffusion model for **structure-to-sequence** design.
   - Iteratively refines random sequences to match a target 3D RNA backbone.
   - Achieves higher sequence recovery than previous inverse folding methods, while providing diverse candidate solutions.

4. **RhoDesign (Deterministic Structure-to-Sequence)**
   - Geometric deep learning approach (using GVP + Transformer) for RNA design.
   - Directly decodes sequences from an input 3D structure, with optional secondary structure constraints.
   - Exhibits state-of-the-art sequence recovery accuracy and structural fidelity in inverse folding tasks.

**Integrated Workflow**  
- **Forward Prediction**: From any RNA sequence, use **RhoFold** to predict 2D/3D structure.  
- **Inverse Design**: Given a target structure, use **RiboDiffusion** or **RhoDesign** to generate novel sequences.  
- **Validation**: Verify designed sequences by folding them back with **RhoFold**, closing the loop.  

This ecosystem enables end-to-end RNA engineering and analysis, creating a powerful platform for synthetic biology, therapeutics, and basic research.

---

## Performance Benchmarks

Thanks to **RNA-FM**’s powerful representation and each project’s specialized architecture, our ecosystem achieves **state-of-the-art** performance:

- **RNA-FM** for Secondary Structure Prediction:  
  - Outperforms classic physics-based and machine learning methods (e.g., LinearFold, SPOT-RNA, UFold) by up to **20–30%** in F1-score on challenging datasets.
  - Particularly robust on long RNAs (>150 nt) and low-homology families.

- **RhoFold** for Tertiary Structure:  
  - Delivers top accuracy on RNA-Puzzles / CASP-type tasks.
  - Predicts 3D structures **within seconds** (single-sequence mode) and integrates MSA for further accuracy gains.
  - Achieved *Nature Methods*–level benchmarks, generalizing to novel RNA families.

- **RiboDiffusion** for Inverse Folding:  
  - A diffusion-based generative approach that surpasses prior methods by **~11–16%** in sequence recovery rate.
  - Provides **tunable diversity** in design, exploring multiple valid sequences for a single target shape.

- **RhoDesign** for Inverse Folding:  
  - A deterministic GVP + Transformer model with **>50%** sequence recovery on standard 3D design benchmarks, nearly double that of older algorithms.
  - Achieves highest structural fidelity (TM-score) among tested methods, validated in *Nature Computational Science*.

Together, these projects set **new standards** for RNA structure prediction and design accuracy, forming an integrated pipeline that can expedite RNA-related research and applications.

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

| Shorthand                                         | Code                                                       | Subject    | Layers | Embed Dim | Max Length | Input  | Token  | Dataset                                  | Description                                                   | Year    | Publisher               |
|---------------------------------------------------|------------------------------------------------------------|------------|--------|----------|-----------|--------|--------|-------------------------------------------|---------------------------------------------------------------|---------|-------------------------|
| **[RNA-FM](https://doi.org/10.48550/arXiv.2204.00300)** | [Yes](https://github.com/ml4bio/RNA-FM)                    | ncRNA      | 12     | 640      | 1024      | Seq    | base   | RNAcentral19 (23M)                        | The first large-scale, general-purpose RNA language model     | 2022.04 | arXiv/bioRxiv          |
| [RNABERT](https://doi.org/10.1093/nargab/lqac012) | [Yes](https://github.com/mana438/RNABERT)                  | ncRNA      | 6      | 120      | 440       | Seq    | base   | RNAcentral (762k) & Rfam14.3 (partial MSA)| Specialized in structural alignment and clustering            | 2022.02 | NAR Genomics Bioinfo   |
| [UNI-RNA](https://doi.org/10.1101/2023.07.11.548588)    | No                                                         | RNA        | 24     | 1280     | ∞         | Seq    | base   | 1B from RNAcentral, nt, GWH               | Larger scale & dataset than RNA-FM; general model             | 2023.07 | bioRxiv                |
| [RNA-MSM](https://doi.org/10.1093/nar/gkad1031)   | [Yes](https://github.com/yikunpku/RNA-MSM)                 | ncRNA      | 12     | 768      | 1024      | MSA    | base   | 4069 families from Rfam14.7               | Uses evolutionary information from MSA                        | 2023.03 | NAR                     |
| [SpliceBERT](https://doi.org/10.1101/2023.01.31.526427) | [Yes](https://github.com/biomedAI/SpliceBERT)             | pre-mRNA   | 6      | 1024     | 512       | Seq    | base   | 2M pre-mRNAs                              | Specialized in RNA splicing                                  | 2023.05 | bioRxiv                |
| [CodonBERT](https://doi.org/10.1101/2023.09.09.556981)  | No                                                         | mRNA CDS   | 12     | 768      | 512×2     | Seq    | codon | 10M mRNAs from NCBI                       | Only for coding sequences; tokenized by codon                 | 2023.09 | bioRxiv                |
| [UTR-LM](https://doi.org/10.1101/2023.10.11.561938)     | [Yes](https://github.com/a96123155/UTR-LM)                 | mRNA 5'UTR | 6      | 128      | ∞         | Seq    | base   | 700k 5'UTRs                               | Targets 5'UTR expression-related tasks                       | 2023.10 | bioRxiv                |
| [3UTRBERT](https://doi.org/10.1101/2023.09.08.556883)   | [Yes](https://github.com/yangyn533/3UTRBERT)               | mRNA 3'UTR | 12     | 768      | 512       | Seq    | k-mer  | 20,362 3'UTRs                             | For 3'UTR-mediated gene regulation tasks                     | 2023.09 | bioRxiv                |
| BigRNA (WIP)                                        | No                                                         | DNA→RNA    | -      | -        | -         | Seq    | -      | Thousands of genome-matched sets          | Tissue-specific expression, splicing, miRNA sites, RBP        | 2023.09 | bioRxiv                |

---

## Citations

If you find our models, code, or pipeline useful in your research, please cite the relevant works.

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

