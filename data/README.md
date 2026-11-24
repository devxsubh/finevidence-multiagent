# Data

`raw/train.json` and `raw/dev.json` are the official FinQA splits pinned to commit
`0f16e2867befa6840783e58be38c9efb9229d742` of https://github.com/czyssrs/FinQA, copied from the
earlier FinEvidence project. Verify them before use:

```bash
shasum -a 256 data/raw/*.json
# 49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6  train.json
# a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51  dev.json
```

The test split is deliberately absent. Data files are Git-ignored; license and provenance are in
`docs/background/dataset-card.md`.
