# Commit B: reveal (written after commit A, before any model result)

drand quicknet round 32846926 (chain 52db9ba7...) was fetched at 2026-10-07T03:36:29Z from two endpoints (api.drand.sh v2, api2.drand.sh v1); both returned the same signature.
Seed = SHA-256(signature) = f9f44cb41feef808ccd3aafd1cc29ec67a165fcd6baadff78a0db34f55733045.

Reproduce: `python3 src/w3/manifest.py --index protocol/index_labels.json --beacon-signature-hex <signature in protocol/drand_round.json> --out manifest.csv`
(verified to reproduce protocol/manifest.csv byte for byte).

- protocol/index_labels.json SHA-256 5a1eda1829ca73d3e91f574962f306624a6439f344293280491b04a8c224eb16 (same as commit A)
- protocol/manifest.csv SHA-256 **a28c1ebddd698806a0f293b3307b4709f251dec086c618513254892a2dcfc206**
- test split found at attempt index 55 (rejections by first violated constraint: {'class 1': 22, 'class 3': 19, 'class 2': 4, 'class 5': 4, 'class 6': 5, 'test row share 0.446': 1}); dev folds at attempt 0.

Test wells (10): 00001, 00004, 00011, 00013, 00015, 00023, 00026, 00035, 00037, 00041
Dev folds: fold 0: 00002, 00008, 00010, 00014, 00016, 00031; fold 1: 00007, 00019, 00022, 00032, 00033, 00040; fold 2: 00012, 00020, 00030, 00036, 00039, 00042; fold 3: 00006, 00009, 00021, 00024, 00034, 00038; fold 4: 00003, 00005, 00025, 00027, 00028, 00029
Support (wells with >=1 episode of >=300 rows) test / dev, by event class:
- class 1: test 00001 | dev 00002,00006
- class 2: test 00011,00013 | dev 00002,00003,00009,00010,00012
- class 3: test 00001 | dev 00014
- class 4: test 00001,00004 | dev 00002,00005,00007,00010,00014
- class 5: test 00015 | dev 00016,00020
- class 6: test 00004 | dev 00002
- class 7: test 00001,00023 | dev 00006,00021,00022,00024
- class 8: test 00026 | dev 00019,00025,00027,00028,00029,00030,00031,00032
- class 9: test 00037,00041 | dev 00014,00016,00040,00042

The manifest is one realisation of the stated constraints; no alternative draw was looked at. The constraint search skipped candidate test sets
that failed the class-coverage, normal-time and row-share constraints (counts above); it never saw any sensor value or model output.
