# Methodology

## Models and training

We trained a conditional flow model and an autoregressive Transformer on antimicrobial peptide sequences published with OmegAMP. Both models use four layers, four attention heads and a hidden dimension of 256. Generation is conditioned on peptide length, charge, hydrophobicity and hydrophobic moment.

The cleaned positive dataset contains 30,449 unique sequences of 8–50 standard amino acids. Similar sequences were grouped before splitting the data into 24,359 training, 3,045 development and 3,045 audit sequences. Training used seed 42, batches of 128 and a learning rate of 0.0003 for 15,000 steps. The retained checkpoints are flow step 14,800 and autoregressive step 800.

## Candidate generation

The sampling recipe combines five initial flow streams, four initial autoregressive streams and 23 production streams. Flow sampling uses Euler or Heun integration; autoregressive sampling varies the temperature. Additional sampling improves coverage of regions represented in the development data.

Production requests 250,000 candidates in batches of 10,000, with a further sampling budget of 50,000. Together with the initial streams, the recorded pool contains 306,328 unique candidates. Sequences identical to the challenge reference are excluded. The stored recipes specify stream order, seeds, sampling settings and batch sizes.

## Library selection

AMPlify estimates antimicrobial activity, and MBC-Attention predicts E. coli MIC. ESM2 embeddings and peptide properties measure how closely a selected library represents the development data.

The hybrid selector first builds a library with a predicted-potency target of 80% and a repair budget of 3,000 replacements. Exchange refinement then improves the selection while requiring at least 88% of library members to have predicted MIC at most 16 µM. Refinement uses up to 60 epochs, seed 42 and four workers.

## Top-100 ranking

Candidates are ranked by the following weighted combination:

```text
0.55 × rank(−log2 predicted MIC)
+ 0.45 × rank(AMPlify score)
− 0.05 × rank(reference similarity)
```

Ties are resolved by increasing predicted MIC, then sequence order. Candidates are added in rank order, subject to reference eligibility and a maximum pairwise normalized Indel similarity of 80% among selected members.

The full library contains 50,000 unique sequences using the 20 standard amino acids, with lengths of 8–50 residues and no exact matches to `data/antibacterial.fasta`. Each top-100 sequence must have a Levenshtein ratio of at most 0.8 against every sequence in that reference. MMseqs2 is an optional diagnostic and is not part of generation or selection.

## Reproducibility

The pipeline regenerates candidate streams, computes predictions and performs selection using the recorded settings. Output hashes are checked against the approved library and ranking. Model predictions guide selection; they are not experimental measurements of antimicrobial activity.
