# AMP Challenge 2027

This submission designs antimicrobial peptides for the [AMP Challenge 2027](https://github.com/szczurek-lab/amp-challenge-2027). It provides a library of 50,000 unique peptides and a ranked selection of 100 members.

Two trained models—a conditional flow model and an autoregressive Transformer—generate candidates. AMPlify and MBC-Attention predict antimicrobial activity and E. coli potency. Selection balances these predictions with sequence diversity and the properties of the development data.

## Run

Use Linux or WSL on x86_64 with a CUDA-capable NVIDIA GPU. The target hardware is an RTX 4060 with 8 GB of GPU memory. Install Git and uv, then run:

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

The tests cover the template's sequence requirements and reproducibility checks. Full reproduction through two fresh GPU runs remains to be verified.

## Method and data

[METHOD.md](METHOD.md) describes generation and ranking. [TRAINING_DATA.md](TRAINING_DATA.md) describes data preparation and splits. [DATA_SOURCES.md](DATA_SOURCES.md) lists sources and citations. Trained generator weights are in `checkpoint/`.

## License

Our code and original generator weights are provided under the [MIT License](LICENSE), a permissive OSI-approved license accepted by the challenge. Third-party materials retain their own licenses; see [NOTICE.md](NOTICE.md).
