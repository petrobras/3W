# Protocol files

Frozen artefacts of protocol 3W-HW v1. The text of the protocol is in the [README of this benchmark](../README.md); the numeric parameters are in `held_out_well/constants.py`.

| file | content |
|---|---|
| `index_labels.json` | one record per real instance of the 3W Dataset 2.0.0 (1,119): path, git blob SHA-1 and size, rows, label runs and operational-state runs, time-axis checks |
| `instances.csv` | the instances in decision-table order: index, path, well, folder, rows, SHA-1 |
| `decisions.npz` | the decision table: instance, row, event class, eligibility and operational state of each of the 962,002 decisions |
| `decisions_arrays.sha256.json` | SHA-256 of each array of `decisions.npz` |
| `manifest.csv` | the 40 wells: role (test or development), development fold, number of files, labelled and normal hours, event classes supported |
| `manifest_info.json` | how the draw went (attempt indices, rejections by rule) and the hashes of the index, the seed and the manifest |
| `drand_round.json` | the committed [drand](https://drand.love) quicknet round and its signature; the seed is the SHA-256 of the signature |
| `COMMIT-A.md`, `COMMIT-B.md` | the commitment written before the draw and the reveal written after it (see below) |
| `kit_summary.json` | counts of instances and decisions |
| `frozen/` | the files whose SHA-256 are listed in `COMMIT-A.md` |

`index_labels.json`, `instances.csv`, `decisions.npz` and `manifest.csv` are derived from the labels of the 3W Dataset 2.0.0 and carry its CC BY 4.0 licence; the other files contain counts, hashes and the public beacon output.

The bytes of these files are hashed and compared. The `.gitattributes` of this folder switches off git's line-ending conversion for them (the repository default is `text=auto`), so a checkout on Windows keeps the hashes.

## The commitment

`COMMIT-A.md` was written before the split existed. It lists the SHA-256 of the label index, of the rule that draws the split and of the label-handling code it imports, and of the protocol text, and it names the drand round (32846926, scheduled for 2026-10-07 03:35:42 UTC) that supplies the randomness. The record lies in the author's earlier repository; it is kept here as it was written, so the paths in it (`src/w3/...`) are those of that repository:

| path in `COMMIT-A.md` | file here |
|---|---|
| `protocol/index_labels.json` | `index_labels.json` |
| `src/w3/manifest.py` | `frozen/manifest.py.txt` |
| `src/w3/labels.py` | `frozen/labels.py.txt` |
| `protocol/PROTOCOL.md` | `frozen/PROTOCOL.md` |

The two code files carry the extension `.txt` so that formatters and linters leave their bytes alone. `held_out_well/manifest.py` and `held_out_well/labels.py` are typeset copies; the tests check that the hashes above match and that the frozen files and the copies give the same `manifest.csv`:

```bash
shasum -a 256 index_labels.json frozen/manifest.py.txt frozen/labels.py.txt frozen/PROTOCOL.md manifest.csv
cp frozen/manifest.py.txt /tmp/manifest.py && cp frozen/labels.py.txt /tmp/labels.py
python /tmp/manifest.py --index index_labels.json \
  --beacon-signature-hex "$(jq -r .signature drand_round.json)" --out /tmp/manifest.csv
cmp /tmp/manifest.csv manifest.csv
```

`COMMIT-B.md` records the signature fetched from two independent drand endpoints and the SHA-256 of the manifest (`a28c1ebd...`).
