# Commit A: split commitment (written BEFORE the manifest exists)

Protocol: 3W-HW v1 (see PROTOCOL.md). Written 2026-10-07T03:31:49Z (local clock; drand latest round at write time: 32846846).

Inputs and code fixed by this commit (SHA-256):
- protocol/index_labels.json  5a1eda1829ca73d3e91f574962f306624a6439f344293280491b04a8c224eb16
- src/w3/manifest.py          94a88b7f140fbdf0d16ebb5a427b4007d5eb0ab579e029782440e9baba54cd14
- src/w3/labels.py            5fc6d4d6c83bbdcb23e6ea154200062f73ede26a6d9821aa9b5eaec94376e1d5
- protocol/PROTOCOL.md        768d5dc7be7771bd69217c4eed271ab2c002072ad12c7752c07898767873b9b5

Source of randomness: drand quicknet (chain hash 52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971), round **32846926**, scheduled for
2026-10-07T03:35:42Z (genesis 1692803367 + (32846926-1) x 3 s). That round did not exist when this file was written.
Seed = SHA-256(bytes of the round's BLS signature) = the beacon's randomness. `manifest.py --beacon-signature-hex <sig>` is the only entry point.
Anyone can re-fetch the round from any drand endpoint and re-run manifest.py to reproduce manifest.csv byte for byte.

Reveal (commit B) adds: the round signature, protocol/manifest.csv and its SHA-256.

Disclosure of what had been looked at before this commit: no model has been trained and no performance number exists. The split inputs are
labels, timestamps, file hashes and process-channel availability only. A smoke kernel printed schema, per-column NaN fractions and label counts for
three instances (dataset/0/WELL-00005_20170814100000, dataset/4/WELL-00002_20140124170136, dataset/9/WELL-00034_20190825004401). Per-file sensor
statistics (index_signals.json) were downloaded but not opened and are not used by manifest.py. Dry runs of manifest.py used throw-away seeds
('dryrun-seed-A/B/C') to test the constraints; their outputs were discarded.
