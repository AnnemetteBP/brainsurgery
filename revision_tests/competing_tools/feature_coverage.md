# Related-system feature coverage

Status: direct Linux comparisons and the scaling study are complete and are
paper evidence. The separate usability/auditability study remains in progress.

This table separates direct, operation-matched comparisons from adjacent
capabilities. It is a claim map, not a count of every feature exposed by any
package. A capability is called **direct** only when BrainSurgery and the named
tool receive the same tensor contract and their outputs are checked by the same
independent oracle.

| Capability | Most relevant comparison | Evidence status | Evidence or disposition |
|---|---|---|---|
| Regex/capture key rewriting | `torch-state-bridge` | Direct comparison complete | R01; 5/5 outputs per tool passed the exact tensor oracle |
| Two-checkpoint weighted merge | MergeKit | Direct comparison complete | M01; 5/5 outputs per tool passed the independent numerical oracle |
| Base-relative task-vector arithmetic | MergeKit | Direct comparison complete | M02; 5/5 outputs per tool passed the independent numerical oracle |
| Block deletion and contiguous reindexing | MergeKit layer selection/slicing | Usability protocol prepared; not a direct tool benchmark | Usability T1 compares BrainSurgery, Python/PyTorch, and an allowed-package condition |
| Tensor slicing and concatenation | MergeKit slicing is adjacent, not identical | Correctness tested; usability protocol prepared | Correctness C06 and usability T2; do not label this a MergeKit head-pruning comparison |
| Mixed-precision conversion and sharded safetensors export | PyTorch/safetensors | Systems comparison complete; usability study separate | All 150/150 scaling attempts passed; usability T3 is also measured separately in the ongoing agent study |
| LoRA merge and dense sharded export | PEFT and MergeKit's adjacent LoRA functionality | Usability protocol prepared; no fixed-tool benchmark | Usability T5; report the package actually selected in condition F rather than implying a MergeKit comparison |
| File-backed/out-of-core execution | Direct PyTorch in-memory baseline; MergeKit has adjacent out-of-core functionality | Systems comparison complete | Scaling compares Python/PyTorch, BrainSurgery in-memory, and BrainSurgery arena; it does not benchmark MergeKit's out-of-core implementation |
| MoE construction/upcycling | MergeKit | Not evaluated | Outside the present revision unless a downstream-quality protocol is added |
| Distributed checkpoint resharding, rank-local state, and optimizer state | Orbax and PyTorch Distributed Checkpoint | Not evaluated | Deferred; exclude from evaluated-capability claims |

## Count that may be reported

- **3 distinct operations** have completed direct comparisons against named
  competing tools: two against MergeKit and one against
  `torch-state-bridge`.
- Those three operations produce **6 tool/case pairings** because each is run
  through BrainSurgery and through its comparator in every repetition.
- The remaining rows are complementary correctness, usability, or systems
  evidence. They must not be added to the direct-comparison count.
- Orbax is related-work positioning, not an executable baseline in this
  revision. No zero, failure, or unsupported score should be assigned to it.

The completed direct study contains 30/30 correct measured outputs: five
repetitions for BrainSurgery and five for the comparator in each of the three
operations. Detailed timings and memory measurements are in
`results/linux_2dbcd50/`.

The corresponding compact LaTeX table is in `feature_coverage.tex`. Update
both files together if an operation is added or its evidence status changes.
