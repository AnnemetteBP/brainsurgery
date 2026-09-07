# EACL 2027 submission-readiness audit

Last audited: 2026-09-07

This audit distinguishes a completed execution from evidence that is ready to
cite in the paper. A result is paper-ready only when its protocol, measured
summary, exact execution revision, interpretation, limitations, and manuscript
placement are all established.

## Current status by revision-plan item

| # | Revision item | Status | Evidence or required action |
|---:|---|---|---|
| 1 | Coding-agent evaluation | **Pending** | Complete both official Claude and Codex cohorts and all manual bookkeeping in `usability_tests/`. Pilots and partial cells are excluded. |
| 2 | Correctness | **Paper-ready** | `correctness/results/paper_table.tex`; 13/13 cases, 679/679 oracle tensor checks, and 664/664 untouched-tensor checks passed across the fixture and real-checkpoint protocols. |
| 3 | Robustness | **Measured; provenance gate open** | Linux 19-case summary and table exist. Preserve the negative R16--R18 findings. The exact recorded execution commit must be made reachable before final citation. |
| 4 | Failure semantics | **Measured; provenance gate open** | The result supports source preservation and safe pre-publication failures, but disproves atomic publication. It shares the robustness provenance gate. |
| 5 | Claims and positioning | **Prepared; manuscript audit pending** | Claim boundaries exist in `claim_boundaries.md`; they have not yet been applied to a complete EACL manuscript. |
| 6 | Reproducibility | **Partial** | Protocols, commands, manifests, run IDs, and compact summaries exist. The completed Linux summaries identify commit `2dbcd505115100f892e906413076ae93b3fcaa16`, which is not currently present in the repository. Behavioral and usability provenance remain pending with those runs. |
| 7 | Competing tools | **Measured; provenance gate open** | Three operation-matched comparisons and the feature-coverage matrix exist. Claims must remain limited to R01, M01, and M02. Recover the recorded execution commit before final citation. |
| 8 | Scaling | **Measured; provenance gate open** | Ten checkpoints through Pythia 12B, 30 model--method pairs, and 150/150 correct measured outputs exist. Report the observed time--memory trade-off and recover the recorded execution commit before final citation. |
| 9 | Behavioral evaluation | **Running/pending** | Only the complete `eacl2027_behavioral_paper_v3` output is eligible. It must contain both original-paper comparisons, every original metric, all ten models, all 70 prompts, and the source/task/language breakdowns. The older v2 result is excluded. |
| 10 | Demo video | **Not done** | Produce a narrated end-to-end workflow only after the final behavior and claims are frozen. |
| 11 | Downstream quality | **Optional; not done** | Do not make downstream-quality claims unless a separately frozen evaluation is completed. Otherwise state the limitation. |
| 12 | RYS/circuit duplication | **Out of scope for this revision** | Do not add it unless all submission-critical work is already complete. |
| 13 | Additional use cases | **Deferred** | Add no breadth-only cases. |
| 14 | Distributed support | **Out of scope** | Explicitly exclude distributed formats, optimizer state, multi-rank execution, and resharding. Indexed safetensors sharding is not distributed evaluation. |
| 15 | Community adoption | **Out of scope** | Do not infer adoption from an agent study. |

## Evidence currently suitable for manuscript integration

The correctness table and its bounded preservation claim are paper-ready now.
The Linux robustness, scaling, and competing-tool measurements are numerically
complete and have paper tables and cautious interpretations, but their shared
recorded execution commit is not reachable in the repository. Treat those
three results as provisional until that provenance link is restored. This is a
provenance defect, not a request to rerun the measurements.

The expanded behavioral result and official usability result do not yet exist
as completed evidence. Do not substitute the old behavioral v2 serialization
check, pilots, or partial usability cells.

## Final acceptance-oriented gates

- Every number in the paper traces to a committed machine-readable summary and
  a reachable execution revision.
- Every table states the evaluated unit, comparison, repetitions or prompt
  count, metric direction, and important limitation.
- Behavioral evidence contains the original PPL, cosine/difference, top-1,
  sequence, and generation measurements rather than only exact tensor checks.
- Usability evidence is reported only after complete cohorts and manual audits.
- Negative robustness findings remain visible in the main interpretation.
- The manuscript makes no universal ease, efficiency, preservation,
  superiority, atomicity, downstream-quality, or distributed-support claim.
- The narrated demo presents the same behavior and limitations as the paper.

