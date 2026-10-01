# Third-party notices

The root MIT License covers our original code and trained generator weights. Third-party code, weights and data retain their applicable terms.

| Component | Source | License or notice |
|---|---|---|
| BATTLE | [battleamp-snakemake](https://github.com/szczurek-lab/battleamp-snakemake/tree/8c659c0cc3d69a260b1865984dbea9d1199f651d) | MIT; see `external-licenses/BATTLE_LICENSE.txt` where included |
| AMPlify | [BattleAMP-amplify](https://github.com/szczurek-lab/BattleAMP-amplify/tree/57bae79d88464b20954ab87cb6840f0f8b582b55) | GPL-3.0; upstream license retained |
| MBC-Attention | [BattleAMP-mbc-attention](https://github.com/szczurek-lab/BattleAMP-mbc-attention/tree/df9e03fbc47714a4cf99dac5cd5fe0c572158a4f) | Wrapper refers to upstream terms; an explicit code/weight redistribution grant has not been established |
| OmegAMP residue-scale tables | [OmegAMP constants](https://github.com/szczurek-lab/OmegAMP/blob/main/project/constants.py) | MIT; see `external-licenses/OMEGAMP_LICENSE.txt` |
| ESM2 | [facebookresearch/esm](https://github.com/facebookresearch/esm) | Upstream MIT notice retained |
| Challenge validator and reference | [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027) | Template BSD-3-Clause notice in `external-licenses/AMP_CHALLENGE_LICENSE.txt` |
| Miniforge bootstrap | [Miniforge 26.7.2-0](https://github.com/conda-forge/miniforge/releases/tag/26.7.2-0) | Upstream installer and package terms apply |

BATTLE and its scoring models are downloaded from pinned public repositories during setup. Source versions and weight checksums are recorded in `assets/battle_pins.json`. Python dependencies are listed in `uv.lock`; scoring environments are recorded under `assets/environments/`.

Dataset attribution is provided in [DATA_SOURCES.md](DATA_SOURCES.md). OmegAMP's software license does not establish redistribution rights for all underlying database records; those terms remain to be confirmed.
