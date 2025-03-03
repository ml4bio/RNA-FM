# RNA-FM: The RNA Foundation Model
[![Pic](./docs/pics/RNA-FM.png)](https://proj.cse.cuhk.edu.hk/rnafm/#/)

[![arXiv](https://img.shields.io/badge/arXiv-2204.00300-b31b1b.svg)](https://arxiv.org/abs/2204.00300)
[![Nature Methods](https://img.shields.io/badge/Nature_Methods-10.1038/s41592--024--02487--0-1f77b4.svg)](https://www.nature.com/articles/s41592-024-02487-0)
[![Nature Computational Science](https://img.shields.io/badge/Nature_Computational_Science-10.1038/s43588--024--00720--6-1f77b4.svg)](https://www.nature.com/articles/s43588-024-00720-6)
[![Bioinformatics](https://img.shields.io/badge/Bioinformatics-10.1093/bioinformatics/btab616-0887f7.svg)](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903)
[![RNA-FM Server](https://img.shields.io/badge/RNA_FM%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
[![RhoFold Server](https://img.shields.io/badge/RhoFold%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/#/)


**Update March 2024**  
1. **Tutorials**  
   - [Tutorial for RNA family clustering and RNA type classification](./tutorials/rna_family-clustering_type-classification/rnafm-tutorial-code.ipynb)  
   - [Tutorial video (in Chinese)](https://www.bilibili.com/video/BV11D4215795/?vd_source=a80c1513b9533b969f95a485ab252511)  
2. **mRNA-FM**  
   - A foundation model pre-trained on coding sequences (CDS) is now released!  
   - `mRNA-FM` can tokenize CDS (3-mer) and represent them with contextual embeddings, benefiting mRNA- and protein-related tasks.  

## Introduction

[**RNA-FM** (RNA Foundation Model)](https://arxiv.org/abs/2204.00300) is a state-of-the-art **pretrained language model for RNA sequences**, serving as the cornerstone of an integrated RNA research ecosystem. Trained on **23+ million non-coding RNA (ncRNA) sequences** via self-supervised learning, RNA-FM captures rich structural and functional signals from RNA sequences *without* requiring experimental labels. Consequently, RNA-FM provides **general-purpose RNA embeddings** that can be applied to a broad range of downstream tasks—such as secondary and tertiary structure prediction, RNA family clustering, and functional RNA analysis.

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

<details open><summary><b>Table of Contents</b></summary>

- [Introduction](#introduction)
- [RNA-FM and Related Tools](#rna-fm-and-related-tools)
  - [RNA-FM (Foundation Model)](#rna-fm-foundation-model)
  - [Downstream Tools](#downstream-tools)
    - [RhoFold (Tertiary Structure Prediction)](#rhofold-tertiary-structure-prediction)
    - [RiboDiffusion (Inverse Folding – Diffusion)](#ribodiffusion-inverse-folding--diffusion)
    - [RhoDesign (Inverse Folding – Deterministic)](#rhodesign-inverse-folding--deterministic)
- [Applications](#applications)
  - [RNA Therapeutics & Drug Design](#rna-therapeutics--drug-design)
  - [Synthetic Biology & Advanced RNA Engineering](#synthetic-biology--advanced-rna-engineering)
  - [Functional Genomics & Biomarker Discovery](#functional-genomics--biomarker-discovery)
  - [Educational & Exploratory Research](#educational--exploratory-research)
- [Setup and Usage](#setup-and-usage)
  - [Setup Environment with Conda](#setup-environment-with-conda)
  - [Quick Start Usage](#quick-start-usage)
  - [Online Server](#online-server)
- [Further Development & Python API](#further-development--python-api)
  - [Usage Examples with the Ecosystem](#usage-examples-with-the-ecosystem)
  - [API Reference](#api-reference)
- [Related RNA Language Models](#related-rna-language-models)
- [Citations](#citations)
- [License](#license)

</details>

---
## RNA-FM and Related Tools

**RNA-FM Ecosystem Components**: Our platform comprises four integrated tools, each addressing a critical step in the RNA analysis and design pipeline:

| Model | Task | Description | Code | Paper                                                                                         |
|-------|------|-------------|------|-----------------------------------------------------------------------------------------------|
| **RNA-FM** | **Foundation Model** (Representation)  | Specialized model for coding sequences using 3-mer tokenization | [GitHub](https://github.com/ml4bio/RNA-FM) | [Nature Methods](https://arxiv.org/abs/2204.00300)|
| **RhoFold** | 3D Structure Prediction | RNA-FM-powered model for sequence-to-structure prediction | [GitHub](https://github.com/ml4bio/RhoFold) | [Nature Methods](https://www.nature.com/articles/s41592-024-02487-0)|
| **RiboDiffusion** | Inverse Folding | Generative diffusion model for structure-to-sequence design | [GitHub](https://github.com/ml4bio/RiboDiffusion) | [Bioinformatics](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i347/7700903) |
| **RhoDesign** | Inverse Folding | Geometric deep learning approach for RNA design | [GitHub](https://github.com/ml4bio/RhoDesign) | [Nature Computational Science](https://www.nature.com/articles/s43588-024-00720-6)|

### RNA-FM (Foundation Model)
- [**RNA-FM (Foundation Model)**](https://github.com/ml4bio/RNA-FM) – A BERT-style Transformer (12 layers, 640 hidden dim) trained on millions of RNA sequences. It learns general-purpose RNA embeddings that encode structural and functional information. RNA-FM provides APIs for embedding extraction and can directly predict base-pairing probabilities for secondary structure.

  <details><summary>Click to expand RNA-FM details</summary>

  [![CUHKServer](https://img.shields.io/badge/CUHK%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
  [![arXiv](https://img.shields.io/badge/arXiv-2204.00300-b31b1b.svg)](https://arxiv.org/abs/2204.00300)
    
    RNA-FM is pre-trained on massive RNA sequence data (RNAcentral) to produce contextual embeddings. These embeddings fuel structure-related tasks (e.g., secondary structure prediction, 3D distance/closeness prediction) and function-related tasks (e.g., UTR function, RNA-protein interaction). The RNA-FM model (12-layer Transformer) is at the core of both pre-training and fine-tuning stages, providing generalizable representations. Downstream, specialized tools (RhoFold, RiboDiffusion, RhoDesign) leverage RNA-FM for end-to-end RNA engineering.
  
  [![RNA-FM Overview](./docs/pics/overview.png)](https://github.com/ml4bio/RNA-FM)

  - **RNA-FM** for Secondary Structure Prediction:
    - Outperforms classic physics-based and machine learning methods (e.g., LinearFold, SPOT-RNA, UFold) by up to **20–30%** in F1-score on challenging datasets.
    - Particularly robust on long RNAs (>150 nt) and low-homology families.
    
  </details>

### Downstream Tools

#### RhoFold (Tertiary Structure Prediction)

- [**RhoFold (Tertiary Structure Prediction)**](https://github.com/ml4bio/RhoFold) – An RNA-FM–powered predictor for RNA 3D structures. Given an RNA sequence, RhoFold rapidly predicts its tertiary structure (3D coordinates in PDB format) along with the secondary structure (CT file) and per-residue confidence scores. It achieves high accuracy on RNA 3D benchmarks by combining RNA-FM embeddings with a structure prediction network, significantly outperforming prior methods in the RNA-Puzzles challenge.

  <details><summary>Click to expand RhoFold details</summary>
  
    [![CUHKServer](https://img.shields.io/badge/CUHK%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/)
    [![Nature Methods](https://img.shields.io/badge/Nature_Methods-10.1038/s41592--024--02487--0-1f77b4.svg)](https://www.nature.com/articles/s41592-024-02487-0)

    RhoFold leverages the powerful embeddings from RNA-FM to revolutionize RNA tertiary structure prediction. By combining deep learning with structural biology principles, RhoFold translates RNA sequences directly into accurate 3D coordinates. The model employs a multi-stage architecture that first converts RNA-FM's contextual representations into distance maps and torsion angles, then assembles these into complete three-dimensional structures. Unlike previous approaches that often struggle with RNA's complex folding landscapes, RhoFold's foundation model approach captures subtle sequence-structure relationships, enabling state-of-the-art performance on challenging benchmarks like RNA-Puzzles. The system works in both single-sequence mode for rapid predictions and can incorporate multiple sequence alignments (MSA) when higher accuracy is needed, making it versatile for various research applications from small RNAs to complex ribozymes and riboswitches.

  [![RhoFlod Overview](https://github.com/ml4bio/RhoFold/raw/main/View.png)](https://github.com/ml4bio/RhoFold)
  - **RhoFold** for Tertiary Structure:
    - Delivers top accuracy on RNA-Puzzles / CASP-type tasks.
    - Predicts 3D structures **within seconds** (single-sequence mode) and integrates MSA for further accuracy gains.
    - Achieved *Nature Methods*–level benchmarks, generalizing to novel RNA families.
    - 
  </details>

#### RiboDiffusion (Inverse Folding – Diffusion)

- [**RiboDiffusion (Inverse Folding – Diffusion)**](https://github.com/ml4bio/RiboDiffusion) – A diffusion-based inverse folding model for RNA design. Starting from a target 3D backbone structure, RiboDiffusion iteratively generates RNA sequences that fold into that shape. This generative approach yields higher sequence recovery (≈11–16% improvement) than previous inverse folding algorithms, while offering tunable diversity in the designed sequences.

  <details><summary>Click to expand RiboDiffusion details</summary>

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

    [![Nature Computational Science](https://img.shields.io/badge/Nature_Computational_Science-10.1038/s43588--024--00720--6-1f77b4.svg)](https://www.nature.com/articles/s43588-024-00720-6)

  RhoDesign introduces a deterministic approach to RNA inverse folding using geometric deep learning. Unlike diffusion-based methods, RhoDesign directly translates 3D structural information into RNA sequences through a specialized architecture combining Graph Vector Perceptrons (GVP) and Transformer networks. This architecture effectively captures both local geometric constraints and global structural patterns in RNA backbones. RhoDesign can incorporate optional secondary structure constraints, allowing researchers to specify certain base-pairing patterns while letting the model optimize the remaining sequence. Benchmark tests demonstrate that RhoDesign achieves remarkable sequence recovery rates exceeding 50% on standard datasets—nearly double the performance of traditional methods. Moreover, the designed sequences exhibit the highest structural fidelity (as measured by TM-score) among current approaches. This combination of accuracy and efficiency makes RhoDesign particularly suitable for precision RNA engineering applications where structural integrity is paramount.

  [![Overview](https://github.com/ml4bio/RhoDesign/raw/main/model_arc.png)](https://github.com/ml4bio/RhoDesign)

  - **RhoDesign** for Inverse Folding:
    - A deterministic GVP + Transformer model with **>50%** sequence recovery on standard 3D design benchmarks, nearly double that of older algorithms.
    - Achieves highest structural fidelity (TM-score) among tested methods, validated in *Nature Computational Science*.

  </details>

**Unified Workflow**: These tools operate in concert to enable end-to-end RNA engineering. For any RNA sequence of interest, one can **predict its structure** (secondary and tertiary) using RNA-FM and RhoFold. Conversely, given a desired RNA structure, one can **design candidate sequences** using RiboDiffusion or RhoDesign (or both for cross-validation). Designed sequences can then be validated by folding them back with RhoFold, closing the loop. This forward-and-inverse design cycle, all powered by RNA-FM embeddings, creates a powerful closed-loop workflow for exploring RNA structure-function space. By seamlessly integrating prediction and design, the RNA-FM ecosystem accelerates the design-build-test paradigm in RNA science, laying the groundwork for breakthroughs in RNA therapeutics, synthetic biology constructs, and our understanding of RNA biology.

---

[//]: # ()
[//]: # (## Applications)

[//]: # ()
[//]: # (The RNA-FM ecosystem unlocks a broad range of applications across biotechnology and research:)

[//]: # ()
[//]: # (- **RNA Therapeutics & Drug Design**)

[//]: # (  - Accelerate the development of RNA-based therapeutics &#40;e.g. mRNA vaccines, siRNA, aptamers, ribozymes&#41; by rapidly evaluating and optimizing candidate sequences *in silico*.)

[//]: # (  - Predict structural stability and functional motifs of therapeutic RNAs, identify drug-binding pockets on viral RNAs, and **design new RNA molecules** with desired functions to reduce experimental trial-and-error.)

[//]: # (- **Synthetic Biology**)

[//]: # (  - Rationally engineer RNA devices such as riboswitches, sensors, and logic circuits. The pipeline can propose sequence variants that achieve a desired structural change in response to stimuli.)

[//]: # (  - Use **RiboDiffusion** or **RhoDesign** to generate novel RNA sequences for a target 3D structure &#40;ensuring the molecule folds into a required shape&#41;, then validate the design by predicting its structure with **RhoFold**. This streamlines the build-test cycle for novel RNA-based components in synthetic biology.)

[//]: # (- **Functional RNA Analysis & Genomics**)

[//]: # (  - Leverage RNA-FM’s embeddings to cluster and classify RNAs by family, discover evolutionary relationships, and annotate non-coding RNAs in genomic data. Complex RNAs with unknown function can be characterized by similarity in embedding space to known classes.)

[//]: # (  - Explore viral RNAs &#40;e.g., SARS-CoV-2 genome elements&#41; to identify conserved structural regions or potential targets for antiviral compounds. The ability to predict structures and interactions helps in understanding RNA viruses and non-coding elements in the genome that could be drug targets or biomarkers.)

[//]: # ()
[//]: # (*&#40;Beyond these, our tools also lower barriers in **education and exploratory research** – students and scientists can easily experiment with RNA structure prediction or design via our user-friendly web server and notebooks, obtaining results in minutes.&#41;*)

[//]: # ()
[//]: # (With the surging interest in RNA technologies &#40;from RNAi therapeutics to RNA vaccines&#41;, the RNA-FM ecosystem provides a timely platform to **streamline RNA discovery and engineering**. Its versatile applications span both academia and industry, accelerating innovation in genomics, drug development, and synthetic biology.)

[//]: # ()
[//]: # (### Integration with RhoFold for RNA Structure Prediction)

[//]: # ()
[//]: # (RNA-FM is a core component of **RhoFold+**, an accurate RNA 3D structure prediction method &#40;analogous to AlphaFold for RNA&#41;. RhoFold+ uses the pretrained RNA-FM as its sequence encoder – transforming input RNA sequences into **evolutionarily and structurally informed embeddings**. These RNA-FM embeddings, together with &#40;optionally&#41; multiple sequence alignment &#40;MSA&#41; features, are fed into RhoFold’s structure prediction network &#40;dubbed Rhoformer&#41;. Rhoformer then iteratively refines the representation and passes it to a geometry-aware structure module &#40;with invariant point attention&#41; that predicts the 3D coordinates of the RNA. In this workflow, RNA-FM provides the language-model prior, effectively telling RhoFold which nucleotide positions are likely paired or structurally important based on learned sequence patterns. This design proved highly successful: **RhoFold+ achieved state-of-the-art accuracy in RNA tertiary structure prediction**, as evidenced by winning performances in RNA-Puzzles and CASP15 RNA categories. It can generate reliable RNA 3D models even **from a single sequence**, overcoming the limited availability of homologous sequences for many RNAs. An online RhoFold server is available for users to predict RNA structures using RNA-FM embeddings under the hood.)

[//]: # ()
[//]: # (**Usage**: To predict an RNA structure with RhoFold+, one provides the RNA sequence &#40;and optionally an MSA&#41;. The RNA-FM model embedded in RhoFold+ will generate sequence embeddings that inform the folding prediction. RhoFold outputs a PDB 3D structure and base-pairing &#40;secondary structure&#41; information. Internally, RNA-FM helps generalize across RNA families – RhoFold+ can even predict structures of novel RNA families that were unseen during training, thanks to RNA-FM’s learned generalizable features. This integration showcases how a foundation model can be fine-tuned &#40;or coupled with task-specific models&#41; to achieve cutting-edge results in a challenging structural biology task.)

[//]: # ()
[//]: # (### Integration with RiboDiffusion and RhoDesign for RNA Design)

[//]: # ()
[//]: # (Beyond structure prediction, RNA-FM also plays a role in **RNA design** – the inverse folding problem of finding sequences that fold into a desired structure. Two complementary approaches in the RNA-FM ecosystem are **RiboDiffusion** and **RhoDesign**:)

[//]: # ()
[//]: # (- **RiboDiffusion** is a generative diffusion model for RNA inverse folding. It conditions a diffusion process on a target 3D backbone and gradually “denoises” into a sequence that is predicted to fold into that structure. While RiboDiffusion’s architecture mainly consists of a geometric GNN and a sequence transformer specific to the diffusion model, it leverages RNA-FM indirectly via RhoFold. In the training phase, RiboDiffusion **augments its training data by using RhoFold &#40;with RNA-FM&#41; to predict structures for many RNA sequences**, obtaining additional structure–sequence pairs beyond the experimentally known ones. These extra samples &#40;RNAcentral sequences folded by RhoFold&#41; improve the model’s ability to generalize. During inference, RiboDiffusion can also use RhoFold to validate that the sequences it generates indeed fold into the desired structure. This two-step workflow &#40;design candidate with RiboDiffusion, then fold with RhoFold&#41; ensures the designed sequences are evaluated and filtered using the RNA-FM-informed structure predictor. RiboDiffusion has demonstrated state-of-the-art performance in RNA inverse folding, outperforming previous methods in success rate and diversity of designs.)

[//]: # ()
[//]: # (- **RhoDesign** is a Transformer encoder–decoder model for structure-conditioned RNA sequence design &#40;a supervised learning approach&#41;. It directly learns to generate RNA sequences given a target secondary and/or tertiary structure input. RhoDesign uses a **geometric vector perceptron &#40;GVP&#41; encoder** to embed the 3D structural context of each nucleotide &#40;nodes in a 3D graph&#41;, and a Transformer decoder to output sequences nucleotide by nucleotide. Importantly, RhoDesign was trained not only on real RNA structures from PDB but also on **predicted structures from RhoFold** to expand the training set. By utilizing RhoFold &#40;powered by RNA-FM&#41; to generate additional training examples, RhoDesign achieved higher sequence recovery rates – i.e., it can redesign known RNA structures with sequences closer to the natural ones. In practice, given a target RNA structure &#40;from experiment or modeling&#41;, RhoDesign’s encoder captures the geometric features &#40;distances, orientations, etc.&#41;, and its decoder produces candidate sequences that are expected to fold into that structure. This method complements RiboDiffusion by offering a fast, one-shot generation based on learned mappings, whereas diffusion provides a stochastic search. Together, these tools illustrate how RNA-FM and its downstream models form a **cohesive workflow**: one can **predict an RNA structure from sequence &#40;RhoFold&#41;** and also **design sequences for a given structure &#40;RhoDesign or RiboDiffusion&#41;**, with RNA-FM providing a unifying knowledge base throughout the process.)

[//]: # ()
[//]: # (In summary, **RNA-FM, RhoFold, RiboDiffusion, and RhoDesign constitute an integrated ecosystem** for RNA biology. Researchers can obtain RNA-FM embeddings for any sequence, use RhoFold+ &#40;with those embeddings&#41; to predict 3D structure, and if needed, use RhoDesign or RiboDiffusion &#40;with RhoFold’s assistance&#41; to inversely design new RNA sequences for a target structure. This end-to-end capability – from sequence to structure to novel sequence – underscores the power of foundation models: RNA-FM generalizes across RNA sequence space, enabling both predictive and generative RNA tasks to reach new levels of accuracy and scale.)

## Applications

Modern biotechnology hinges on precise RNA analysis and design, making **RNA-FM** and its ecosystem highly valuable in both scientific and commercial settings. By enabling rapid, accurate structure prediction, advanced RNA design, and functional insights, our integrated platform serves as an indispensable resource for multiple applications:

### RNA Therapeutics & Drug Design

In the booming field of RNA therapeutics—spanning mRNA vaccines, siRNA treatments, and aptamer-based diagnostics—speed and accuracy are critical. **RNA-FM** provides:
- **Accelerated R&D**: Rapidly predict RNA structural stability, identify active motifs, and optimize mRNA constructs to reduce experimental failures.
- **Target Discovery & Validation**: Model viral RNAs (e.g., SARS-CoV-2) or non-coding RNAs to pinpoint druggable structural pockets, expediting lead identification.
- **Streamlined Production Pipelines**: De-risk costly development by screening candidate RNA therapeutics *in silico* before in vitro or in vivo testing.

### Synthetic Biology & Advanced RNA Engineering

Bioengineering initiatives increasingly rely on RNA to build novel biosensors, gene regulators, and molecular circuits:
- **Tailored RNA Components**: Use **RiboDiffusion** or **RhoDesign** to create bespoke riboswitches, regulatory RNAs, or catalytic motifs. Incorporate new functional domains while preserving structural integrity.
- **Design-Build-Test Cycle**: Validate newly designed constructs *in silico* with **RhoFold**, reducing iteration time and project costs. This rapid prototyping fosters high-impact synthetic biology discoveries.

### Functional Genomics & Biomarker Discovery

As large-scale RNA sequencing continues to transform genomics, **RNA-FM** plays a pivotal role in:
- **Automated RNA Annotation**: Cluster and classify RNAs by leveraging advanced embeddings, revealing novel RNA families, regulatory elements, or potential biomarkers in massive transcriptome datasets.
- **Predictive Functional Profiling**: Assess expression stability, splicing variants, and RNA-protein interactions to better understand disease mechanisms and identify diagnostic targets.
- **Therapeutic Target Prioritization**: Combine structure predictions with omics data to prioritize RNA candidates for follow-up validation in oncology, virology, or precision medicine applications.

### Educational & Exploratory Research

**RNA-FM** not only benefits industry partners but also serves as a powerful academic research tool:
- **User-Friendly Web Interface**: Gain immediate structural or functional insights without local computing resources, making advanced modeling accessible to educators and students.
- **Open-Source Community**: Join a growing ecosystem of developers building specialized plugins and analysis tools, fostering transparency and collaborative innovation in RNA research.
- **Accelerated Discovery**: Quickly test hypotheses on RNA folding or function, enabling researchers to iterate faster and publish breakthroughs in structural and molecular biology.

By delivering unprecedented accuracy in RNA structure prediction and design, **RNA-FM** and its companion tools provide a strategic advantage in fields where RNA’s central role is rapidly expanding. This integrated system cuts both time and cost associated with trial-and-error experimentation, thereby unlocking significant value for investors, biotech developers, and academic institutions alike.

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

   - For **mRNA-FM**, ensure that your input RNA sequences have lengths multiple of 3 (codons) and place the specialized weights for *mRNA-FM* in the same `pretrained` folder.


### Quick Start Usage


#### 1. Embedding Generation


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

#### 2. RNA Secondary Structure Prediction

```bash
python launch/predict.py \
    --config="pretrained/ss_prediction.yml" \
    --data_path="./data/examples/example.fasta" \
    --save_dir="./results" \
    --save_frequency 1
```

RNA-FM will output base-pair probability matrices (`.npy`) and secondary structures (`.ct`) to `./results/r-ss`.

[//]: # (### 3. Online Server <a name="Server"></a>)
### Online Server
[![RNA-FM Server](https://img.shields.io/badge/RNA_FM%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/rnafm/#/)
[![RhoFold Server](https://img.shields.io/badge/RhoFold%20Server-Running-green.svg)](https://proj.cse.cuhk.edu.hk/aihlab/RhoFold/#/)

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

Generates 5 new sequences matching `R1107.pdb`. Output FASTA files are placed in `exp_inf/fasta/`.

#### RhoDesign (Structure → Sequence)

```bash
cd RhoDesign
python src/inference.py \
    --pdb ../example/2zh6_B.pdb \
    --ss ../example/2zh6_B.npy \
    --save ../example/
```

Produces a designed sequence that folds into the target 3D shape, leveraging geometry-based encoding. Adjust temperature or other parameters for more diversity or fidelity.

</details>

### API Reference

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

