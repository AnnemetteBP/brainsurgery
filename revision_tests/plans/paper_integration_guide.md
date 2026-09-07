# Exact paper-integration guide

This is the insertion plan for the EACL 2027 BrainSurgery demo paper. It
distinguishes main-paper evidence from appendix detail and names the exact
repository artifact to use. Do not create one appendix per experiment.

## Recommended structure

Use one main-paper section titled **Evaluation** and two appendices:

1. **Appendix A: Automated Evaluation Protocols and Detailed Results**
2. **Appendix B: Coding-Agent Usability and Auditability Study**

Appendix A contains correctness, behavioral equivalence, robustness, scaling,
and competing-tool details as subsections. These belong together because they
are automated checkpoint/evaluation protocols. Appendix B remains separate
because it has a different study design, units of analysis, manual review, and
completion criteria.

The qualitative related-system capability matrix belongs with Related Work,
not as another experimental appendix. Distributed execution, resharding, and
optimizer state have no results and therefore receive only a scope limitation.

## Main-paper Evaluation section

### 1. Correctness and preservation

Use in the main paper:

- Table: `revision_tests/correctness/results/paper_table.tex`
- Prose: the `Independent correctness and preservation` paragraph in
  `revision_tests/plans/paper_evidence_text.tex`

This is the foundational evidence that declared transformations produced the
expected tensor state and left tensors outside their write-set unchanged. The
two correctness rows belong in the same table because they are distinct
protocols: controlled fixtures and pinned real checkpoints. Do not split them
into separate appendices.

### 2. Behavioral equivalence and preservation

This is a core main-paper result, but it is **not yet available**. After the
complete Linux/CUDA v3 run, use:

- Table: `revision_tests/behavioral/results/<v3_run_id>/table.tex`
- Prose: `revision_tests/behavioral/results/<v3_run_id>/paper_text.tex`
- Auditable values: `revision_tests/behavioral/results/<v3_run_id>/evidence.json`

The generated table deliberately combines the two previous-paper comparisons:

1. BrainSurgery versus separate Python/PyTorch implementation: PPL ratio,
   final-token cosine, absolute logit difference, and top-1 agreement.
2. Original versus forward--backward restored checkpoint: full-sequence
   cosine/difference and exact/approximate output agreement.

They belong in one behavioral subsection and one table because together they
answer whether the declarative implementation matches the imperative
implementation and whether a reversible round trip preserves the original
model. Do not use `behavioral/results/linux_99693f2/summary.json` for this
claim; it is only an older multiply-by-one engineering record.

### 3. Comparison with overlapping tools

Use in the main paper:

- Table: `revision_tests/competing_tools/results/paper_table.tex`
- Prose: the `Comparison with overlapping tools` paragraph in
  `revision_tests/plans/paper_evidence_text.tex`

Keep this table in the main paper because direct comparison with overlapping
tools was an explicit reviewer concern. State that it covers only R01, M01,
and M02. The result does not establish a general tool ranking or usability
advantage.

### 4. Scaling and systems behavior

In the main paper, use the concise scaling paragraph from
`revision_tests/plans/paper_evidence_text.tex`. Put the full 30-row table in
Appendix A:

- Full table: `revision_tests/scaling/results/linux_2dbcd50/paper_table.tex`
- Full machine-readable summary:
  `revision_tests/scaling/results/linux_2dbcd50/summary.json`

The main text must state the four-point Pythia conclusion and the 12B numbers:
Python/PyTorch was faster, while BrainSurgery's in-memory provider used less
peak RSS at 12B. Describe this as a measured time--memory trade-off. GPT-2,
OLMo, and Qwen are architecture/storage checks and must not be pooled into the
Pythia scaling curve.

### 5. Robustness and failure semantics

In the main paper, use the robustness paragraph from
`revision_tests/plans/paper_evidence_text.tex`, including the negative result
that three mid-save cases left partial or mixed destinations. Put the compact
table and protocol detail in Appendix A:

- Table: `revision_tests/robustness/results/paper_table.tex`
- Detailed prose: `revision_tests/robustness/results/paper_text.md`

Report the Linux count as 19 cases. The macOS execution is a replication of
the same cases, not another sample, so do not report 38.

### 6. Coding-agent usability and auditability

Do not insert numerical results until both official Claude and Codex cohorts
and their manual audits are complete. When complete, the main paper should
contain the primary success/error/review findings and Appendix B should contain
the task specifications, isolation procedure, model/subscription conditions,
manual rubric, exclusions, per-cell results, and secondary token/time/size
measurements. Do not combine these observations with the automated correctness
counts.

## Appendix A: Automated Evaluation Protocols and Detailed Results

Use this exact subsection order:

### A.1 Correctness oracle and preservation scope

Include fixture construction, write-set definition, corruption controls,
checkpoint revisions, and the limitation that custom safetensors header
metadata and arbitrary sidecars are outside the tensor-state claim. The main
correctness table need not be duplicated here.

### A.2 Behavioral prompt suite and measurement definitions

After v3 completes, document the 30 Belebele, 30 MMLU, and 10 HumanEval
selection; six language varieties; deterministic decoding; model revisions;
PPL definition; final-token and full-sequence cosine/difference definitions;
generation comparison; and source/task/language breakdowns. Put detailed
per-model or per-source results here if the generated main table is too large.

### A.3 Robustness and failure semantics

Insert `revision_tests/robustness/results/paper_table.tex`. Describe the 19
cases and the independent file/tensor auditor. Preserve R16--R18 as negative
findings and give the fresh-destination plus post-save-validation guidance.

### A.4 Scaling protocol and complete results

Insert `revision_tests/scaling/results/linux_2dbcd50/paper_table.tex`. State
that measurements are Linux, single-process, CPU/I/O measurements with five
correctness-validated repetitions. Include hardware/software identifiers from
the environment record captured automatically by the scaling runner at
`log/revision_tests/eacl2027_scaling_linux_2dbcd50/scaling/environment.json`.
It contains the platform/kernel string, CPU counts and affinity, RAM, disk and
filesystem information, Python and package versions, GPU inventory, Git
commit, worker count, sampling interval, and workload note. Explicitly exclude
GPU speed, distributed execution, optimizer state, and resharding.

### A.5 Competing-operation definitions

Define R01, M01, and M02 so readers can verify that both tools performed the
same operation. If space permits, include
`revision_tests/competing_tools/feature_coverage.tex` here; otherwise summarize
it in Related Work. Mark MergeKit slicing, PEFT, Orbax, and PyTorch DCP as
adjacent systems where no like-for-like executable benchmark was run.

The competing-tool runner separately captured its full environment before any
measurement at
`log/revision_tests/eacl2027_competing_linux_2dbcd50/competing_tools/environment.json`.
That record includes platform/kernel, CPU counts and affinity, RAM, disk and
filesystem, Git state, workload controls, and complete package snapshots for
the isolated BrainSurgery, MergeKit, and torch-state-bridge environments.

These records—not UCloud job-page metadata—are the authoritative source for
the shared **Linux experimental environment** paragraph at the start of
Appendix A. They are private raw records because they contain a hostname and
absolute paths; copy only non-identifying hardware, software-version, and
filesystem fields into the anonymous paper. No completed evaluation must be
rerun to obtain this information.

The robustness runner also captured its Linux environment at
`log/revision_tests/eacl2027_robustness_linux_2dbcd50/robustness/environment.json`.
The two macOS correctness environment records are already committed inside
their respective `revision_tests/correctness/results/` directories.

## Appendix B: Coding-Agent Usability and Auditability Study

Keep this appendix pending until the official study closes. Use these
subsections:

1. **B.1 Study design and hypotheses**
2. **B.2 Tasks, conditions, and documentation pack**
3. **B.3 Automated scoring and manual audit rubric**
4. **B.4 Complete results and exclusions**
5. **B.5 Threats to validity**

Do not describe pilots or incomplete cells as study evidence.

## Related Work placement

Use `revision_tests/competing_tools/feature_coverage.tex` either at the end of
Related Work or in Appendix A.5, but not in both places. It is a qualitative
coverage map, not a performance table. The actual performance comparison is
`revision_tests/competing_tools/results/paper_table.tex`.

## Evidence that must not be inserted

- The old behavioral v2 multiply-by-one result as an answer to the behavioral
  reviewer concern.
- Any macOS scaling timing or memory table.
- The removed macOS competing-tool preflight table.
- Incomplete Claude or Codex usability cells.
- Claims about distributed execution, resharding, optimizer-state support,
  atomic output publication, universal information preservation, universal
  ease of use, or universal performance superiority.

## Minimal main-paper order

If page space is severe, keep this order:

1. correctness table and result paragraph;
2. completed v3 behavioral table and result paragraph;
3. competing-tool table and narrow interpretation;
4. scaling and robustness result paragraphs with their full tables in
   Appendix A;
5. completed usability primary table, with study detail in Appendix B.
