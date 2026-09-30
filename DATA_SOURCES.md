# Data sources

| Data or resource | Source | Use |
|---|---|---|
| Positive AMP sequences | [OmegAMP AMPs.fasta](https://github.com/szczurek-lab/OmegAMP/blob/main/data/generative-model-data/AMPs.fasta); DRAMP, dbAMP and AMPScanner | Flow and autoregressive model training; development and audit splits |
| Curated negative sequences | [OmegAMP curated-Non-AMPs.fasta](https://github.com/szczurek-lab/OmegAMP/blob/main/data/activity-data/curated-Non-AMPs.fasta); DBAASP record IDs | Local pilot classifier |
| Antibacterial reference | [Challenge antibacterial.fasta](https://github.com/szczurek-lab/amp-challenge-2027/blob/main/data/antibacterial.fasta); MarLys IDs and source-database tags | Exact-match exclusion and top-100 similarity checks |
| Residue-scale tables | [OmegAMP constants](https://github.com/szczurek-lab/OmegAMP/blob/main/project/constants.py) | Numerical representation of amino acids |
| AMPlify training data | APD3, DADP and UniProtKB/Swiss-Prot; [published datasets](https://doi.org/10.5281/zenodo.7320306) | External pretrained activity model |
| MBC-Attention training data | [DBAASP v3-derived E. coli data](https://github.com/jieluyan/MBC-Attention/tree/main/data), collected August 2021 | External pretrained MIC model |
| ESM2 training data | [UniRef-derived UR50/D 2021_04](https://github.com/facebookresearch/esm#available-models-and-datasets) | External pretrained sequence embeddings |

## Citations

- OmegAMP: Soares et al., *OmegAMP: Targeted AMP Discovery via Biologically Informed Generation*, TMLR (2026), https://openreview.net/forum?id=hAq3XLZ9ex . Earlier paper dataset description: https://arxiv.org/html/2504.17247v1#A5 . Repository: https://github.com/szczurek-lab/OmegAMP .
- DRAMP: Shi et al., *DRAMP 3.0*, Nucleic Acids Research (2022), https://doi.org/10.1093/nar/gkab651 ; database https://dramp.cpu-bioinfor.org/ .
- dbAMP: Jhong et al., *dbAMP 2.0*, Nucleic Acids Research (2022), https://doi.org/10.1093/nar/gkab1080 ; database https://awi.cuhk.edu.cn/dbAMP/ .
- AMPScanner: Veltri, Kamath and Shehu, *Deep learning improves antimicrobial peptide recognition*, Bioinformatics (2018), https://doi.org/10.1093/bioinformatics/bty179 .
- DBAASP: Pirtskhalava et al., *DBAASP v3*, Nucleic Acids Research (2021), https://doi.org/10.1093/nar/gkaa991 ; https://dbaasp.org/ .
- AMPlify: Li et al., BMC Genomics (2022), https://doi.org/10.1186/s12864-022-08310-4 ; training-data disclosure: https://pmc.ncbi.nlm.nih.gov/articles/PMC9896668/ .
- MBC-Attention: Yan et al., mSystems (2023), https://doi.org/10.1128/msystems.00345-23 .
- ESM2: https://github.com/facebookresearch/esm#available-models-and-datasets ; Lin et al., Science (2023), https://doi.org/10.1126/science.ade2574 .
- MarLys dataset: https://data.mendeley.com/datasets/w4hb5grjwb/3 ; associated software https://github.com/bmcode00/marlys-amp .
