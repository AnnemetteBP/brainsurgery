# Competing-tool comparison

Reported Linux run: `eacl2027_competing_linux_2dbcd50` at commit
`2dbcd505115100f892e906413076ae93b3fcaa16`. All automated eligibility gates
passed.

| **Case** | **Tool** | **Correct runs ↑** | **Median wall (s) ↓** | **Median peak RSS (MiB) ↓** | **Output (MiB)** | **Spec lines** |
|---|---|---:|---:|---:|---:|---:|
| R01 | BrainSurgery | 5/5 | 5.042 | 1785.7 | 522.7 | 7 |
| R01 | `torch-state-bridge` | 5/5 | 1.478 | 1571.9 | 522.7 | 2 |
| M01 | BrainSurgery | 5/5 | 5.292 | 2166.7 | 474.7 | 17 |
| M01 | MergeKit | 5/5 | 4.887 | 2890.4 | 474.7 | 11 |
| M02 | BrainSurgery | 5/5 | 5.480 | 3115.1 | 474.7 | 32 |
| M02 | MergeKit | 5/5 | 5.289 | 3895.3 | 474.7 | 13 |

| **Case** | **Competitor** | **BrainSurgery / competitor median wall ratio ↓** |
|---|---|---:|
| R01 | `torch-state-bridge` | 3.412 |
| M01 | MergeKit | 1.083 |
| M02 | MergeKit | 1.036 |

All 30/30 measured outputs passed the independent oracle. Timings include
process startup, checkpoint loading, transformation, and output saving.
Specification lines are descriptive and are not a usability measure.
