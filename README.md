# Active-matter reproductions

Private working collection of the supplied Ising and ExpTM reproductions.

| Folder | Entry point | Preserved work |
| --- | --- | --- |
| `ising/` | `thermomaps_ising.ipynb` | Executed notebook and unmodified upstream source |
| `exptm/` | `rerun_one_sample.py` | Author-checkpoint replay, original reports and generated samples |

Use Python 3.11 and **a separate virtual environment for each folder**. Both projects contain a different package named `tm`; do not import both in one kernel. macOS uses CPU by default. Installation and running instructions are provided in the accompanying chat.

The Ising notebook includes training; no trained Ising checkpoint was supplied. ExpTM downloads three hash-verified upstream assets (~125 MB) before replay. Existing records are preliminary work, not a verified reproduction of all published figures or evidence of stationary phase separation.

Author code retains its provenance; this collection does not assert a new license over third-party work. See `PROVENANCE.json` and `exptm/SOURCE_PIN_AND_LICENSE.json`.
