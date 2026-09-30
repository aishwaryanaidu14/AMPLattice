# AMP Challenge 2027

This submission designs antimicrobial peptides for the [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027). It provides a library of 50,000 unique peptides and a ranked selection of 100 members.

Two trained models—a conditional flow model and an autoregressive Transformer—generate candidates. AMPlify and MBC-Attention predict antimicrobial activity and E. coli potency. Selection balances these predictions with sequence diversity and the properties of the development data.

### Abstract

We use a hybrid generation and selection pipeline with two independently trained generators: a conditional flow model and an autoregressive Transformer. Both models have four layers, four attention heads and hidden dimension 256, and are trained on the positive AMP sequence collection distributed with OmegAMP [1]. Generation is conditioned on peptide length, net charge, hydrophobicity and hydrophobic moment.

Candidate generation combines multiple flow and autoregressive sampling streams. Flow samples use Euler or Heun integration, while autoregressive sampling uses different temperatures and seeds. The resulting candidates are scored with AMPlify for predicted antimicrobial activity [5] and MBC-Attention for predicted *E. coli* MIC [6]. ESM2 embeddings [7] and physicochemical properties are then used to preserve diversity and representation of AMP-like sequence space during library construction.

The final 50,000-peptide library balances predicted potency with sequence and embedding-space coverage. The top 100 are ranked separately using predicted MIC, AMPlify activity and similarity to the antibacterial reference, with additional novelty and pairwise-diversity constraints.

### Data description

The generators are trained on the positive peptide FASTA released with OmegAMP (`data/generative-model-data/AMPs.fasta`) [1]. According to the accompanying data provenance, these AMP sequences originate from DRAMP [2], dbAMP [3] and AMPScanner [4].

We retain only sequences containing the 20 standard proteinogenic amino acids, restrict length to 8–50 residues and remove duplicates. This produces 30,449 positive sequences. Similar sequences are grouped at ≥80% normalized Indel similarity before splitting, giving 24,359 training, 3,045 development and 3,045 audit sequences. Only the positive training sequences are used to train the generators.

Curated non-AMP sequences distributed with OmegAMP (`curated-Non-AMPs.fasta`), linked to DBAASP records [8], are used only for local pilot classification experiments and not as generator training targets. Residue-level physicochemical features use the amino-acid scales distributed in the OmegAMP implementation [1].

For candidate evaluation, we use externally pretrained AMPlify [5], MBC-Attention [6] and ESM2 [7]. The competition-provided `data/antibacterial.fasta`, containing MarLys identifiers and source-database annotations [9], is used for exact-match exclusion and top-100 similarity checks.

### Top candidates selection procedure (for Wet-Lab Track only)

The top-100 list is ranked separately from construction of the 50,000-member library using:

```text
0.55 × rank(−log2 predicted MIC)
+ 0.45 × rank(AMPlify score)
− 0.05 × rank(reference similarity)
```

MBC-Attention provides predicted *E. coli* MIC [6], while AMPlify provides the predicted antimicrobial-activity score [5]. Lower predicted MIC and higher AMPlify activity are preferred, while the reference-similarity term favors novelty.

Ties are resolved by lower predicted MIC and then deterministically by sequence order. Candidates are selected in rank order subject to:

- a Levenshtein similarity ratio of at most 0.8 against every sequence in the competition antibacterial reference; and
- a maximum pairwise normalized Indel similarity of 80% among selected top-100 peptides.

This produces a top-100 set intended to combine predicted antimicrobial potency, novelty and sequence diversity.


## Run

Use Linux or WSL on x86_64 with a CUDA-capable NVIDIA GPU. It was tested on hardware RTX 4060 with 8 GB of GPU memory. Install Git and uv, then run:

```bash
uv sync
uv run generate
```

Python is specified in `.python-version`, and dependencies are recorded in `uv.lock`. On first use, generation downloads the pinned BATTLE repositories and installs their scoring dependencies in separate Conda environments. This setup is automatic and requires internet access.

The output files are:

| File | Contents |
|---|---|
| `generate/library.fasta` | 50,000 unique peptides, each 8–50 residues long |
| `generate/top.fasta` | 100 ranked members of the library |

All arguments have defaults. The default seed is 42. Generation runs model inference and selection, then verifies the resulting file hashes before writing the outputs. Existing output files are removed when generation starts.

Run the tests with:

```bash
uv run python -m unittest discover -s tests
```

The tests cover the template's sequence requirements and reproducibility checks. 

## Method and data

[METHOD.md](METHOD.md) describes generation and ranking. [TRAINING_DATA.md](TRAINING_DATA.md) describes data preparation and splits. [DATA_SOURCES.md](DATA_SOURCES.md) lists sources and citations. Trained generator weights are in `checkpoint/`.

### References

[1] Soares et al., **OmegAMP: Targeted AMP Discovery via Biologically Informed Generation**, TMLR, 2026.  
https://openreview.net/forum?id=hAq3XLZ9ex  
Repository and training sequence files: https://github.com/szczurek-lab/OmegAMP

[2] Shi et al., **DRAMP 3.0**, *Nucleic Acids Research*, 2022.  
https://doi.org/10.1093/nar/gkab651  
https://dramp.cpu-bioinfor.org/

[3] Jhong et al., **dbAMP 2.0**, *Nucleic Acids Research*, 2022.  
https://doi.org/10.1093/nar/gkab1080  
https://awi.cuhk.edu.cn/dbAMP/

[4] Veltri, Kamath and Shehu, **Deep learning improves antimicrobial peptide recognition**, *Bioinformatics*, 2018.  
https://doi.org/10.1093/bioinformatics/bty179

[5] Li et al., **AMPlify**, *BMC Genomics*, 2022.  
https://doi.org/10.1186/s12864-022-08310-4  
Training datasets: https://doi.org/10.5281/zenodo.7320306

[6] Yan et al., **MBC-Attention**, *mSystems*, 2023.  
https://doi.org/10.1128/msystems.00345-23  
Training data: https://github.com/jieluyan/MBC-Attention/tree/main/data

[7] Lin et al., **Evolutionary-scale prediction of atomic-level protein structure with a language model**, *Science*, 2023.  
https://doi.org/10.1126/science.ade2574  
ESM2 models and datasets: https://github.com/facebookresearch/esm#available-models-and-datasets

[8] Pirtskhalava et al., **DBAASP v3**, *Nucleic Acids Research*, 2021.  
https://doi.org/10.1093/nar/gkaa991  
https://dbaasp.org/

[9] **MarLys AMP dataset**, Mendeley Data, Version 3.  
https://data.mendeley.com/datasets/w4hb5grjwb/3

## License

Our code and original generator weights are provided under the [MIT License](LICENSE), a permissive OSI-approved license accepted by the challenge. Third-party materials retain their own licenses; see [NOTICE.md](NOTICE.md).

