from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent / "figures"


def _panel(ax, x: float, y: float, w: float, h: float, title: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.4,
            edgecolor="#1f2937",
            facecolor="#fffdf9",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h - 0.085),
            w,
            0.085,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.043,
        title,
        ha="center",
        va="center",
        fontsize=13,
        color="white",
        fontweight="bold",
        family="DejaVu Serif",
    )


def _code_block(ax, x: float, y: float, lines: list[str], *, size: int = 9) -> None:
    step = 0.032
    for i, line in enumerate(lines):
        ax.text(
            x,
            y - i * step,
            line,
            ha="left",
            va="top",
            fontsize=size,
            family="DejaVu Sans Mono",
            color="#17202a",
        )


def _callout(ax, x: float, y: float, w: float, h: float, text: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=1.0,
            edgecolor="#9a6b2f",
            facecolor="#f3e3cf",
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.2,
        family="DejaVu Serif",
        color="#17202a",
        linespacing=1.2,
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_comparison_figure() -> None:
    fig = plt.figure(figsize=(15.5, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f1e8")

    ax.text(
        0.5,
        0.965,
        "Dense-to-Expert-MoE Upcycling: Imperative Baseline vs Declarative BrainSurgery",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.93,
        "Same conversion semantics, but explicit checkpoint surgery and validation replace handwritten control flow.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    _panel(ax, 0.035, 0.13, 0.28, 0.73, "A. Reference Python Script", "#9f2f21")
    _panel(ax, 0.36, 0.13, 0.28, 0.73, "B. BrainSurgery YAML Plan", "#155e63")
    _panel(ax, 0.685, 0.13, 0.28, 0.73, "C. Built-In Validation", "#2f5f2f")

    _code_block(
        ax,
        0.055,
        0.755,
        [
            'moe_to_dense_mapping = {',
            '  "feed_forward_moe.experts.mlp.w1": ...',
            '  "feed_forward_moe.experts.mlp.w2": ...',
            '  "attention.w_q.weight": ...',
            "}",
            "",
            "for expert, path in enumerate(dense_paths):",
            "  dense_state_dict = load_state_dict(path)",
            "  for key in list(moe_state_dict.keys()):",
            "    if any(pattern in key for pattern in ...):",
            "      dense_key = key.replace(...)",
            '      if "expert" in key or "router" in key:',
            "        moe_state_dict[key][...] =",
            "          dense_state_dict[dense_key].T",
            "      ...",
        ],
    )
    _callout(
        ax,
        0.05,
        0.145,
        0.25,
        0.07,
        "Control flow, mapping, mutation,\nand output writing are intertwined.",
    )

    _code_block(
        ax,
        0.38,
        0.755,
        [
            "transforms:",
            "  - assert: { exists: m0::model.embed_tokens.weight }",
            "  - assert:",
            "      equal:",
            r"        left:  m0::model.layers\.(\d+)\.mlp...",
            r"        right: m1::model.layers.\1.mlp...",
            "  - copy:",
            r"      from: m0::model.layers\.(\d+)\.mlp...",
            r"      to:   m0::model.layers.\1.mlp.experts.0...",
            "  - copy:",
            r"      from: m1::model.layers\.(\d+)\.mlp...",
            r"      to:   m0::model.layers.\1.mlp.experts.1...",
            "  - fill:",
            r"      to: m0::model.layers.\1.mlp.gate.weight",
            "      mode: constant",
            "      value: 0",
            r"  - delete: { target: m0::model.layers\.(\d+)\.mlp... }",
        ],
    )
    _callout(
        ax,
        0.375,
        0.145,
        0.25,
        0.07,
        "Assertions, surgery, and validation are\nexplicit, reviewable, and reusable.",
    )

    _code_block(
        ax,
        0.705,
        0.755,
        [
            "inputs:",
            "  - yaml::olmo_1b_0724_hf_dense_moe_demo",
            "  - ref::olmo_1b_0724_hf_dense_moe_reference",
            "",
            "transforms:",
            "  - diff: { mode: aliases, left_alias: ref,",
            "            right_alias: yaml }",
            "",
            "Missing on left:",
            "  (none)",
            "Missing on right:",
            "  (none)",
            "Differing:",
            "  (none)",
            "No differences found.",
        ],
    )
    _callout(
        ax,
        0.7,
        0.145,
        0.25,
        0.07,
        "The declarative plan matches the\nindependent reference implementation.",
    )

    ax.add_patch(
        FancyArrowPatch((0.315, 0.495), (0.36, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )
    ax.add_patch(
        FancyArrowPatch((0.64, 0.495), (0.685, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )

    ax.text(
        0.5,
        0.07,
        "Dense checkpoint A + Dense checkpoint B  ->  conversion  ->  MoE-style checkpoint  ->  BrainSurgery diff",
        ha="center",
        va="center",
        fontsize=12,
        family="DejaVu Serif",
        color="#17202a",
    )

    _save(fig, "olmo_1b_0724_comparison")


def make_pipeline_figure() -> None:
    fig = plt.figure(figsize=(13.5, 4.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.93,
        "Validated BrainSurgery Workflow for Dense-to-Expert-MoE Upcycling",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )

    boxes = [
        (0.04, 0.42, 0.17, 0.2, "#e8eef9", "Dense\ncheckpoint A"),
        (0.04, 0.14, 0.17, 0.2, "#e8eef9", "Dense\ncheckpoint B"),
        (0.29, 0.28, 0.19, 0.2, "#dff3f1", "BrainSurgery\nYAML plan"),
        (0.55, 0.42, 0.17, 0.2, "#eef7e8", "Converted\nMoE checkpoint"),
        (0.55, 0.14, 0.17, 0.2, "#f7efe8", "Reference\nPython output"),
        (0.80, 0.28, 0.16, 0.2, "#edf7ed", "Diff result:\nNo differences\nfound"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#1f2937",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            family="DejaVu Serif",
            color="#17202a",
            fontweight="bold",
            linespacing=1.15,
        )

    arrows = [
        ((0.21, 0.52), (0.29, 0.40)),
        ((0.21, 0.24), (0.29, 0.36)),
        ((0.48, 0.39), (0.55, 0.52)),
        ((0.21, 0.24), (0.55, 0.24)),
        ((0.72, 0.52), (0.80, 0.39)),
        ((0.72, 0.24), (0.80, 0.37)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#7c3f00")
        )

    ax.text(
        0.5,
        0.06,
        "BrainSurgery externalizes checkpoint surgery as a declarative, executable,\nand verifiable artifact.",
        ha="center",
        va="center",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
        linespacing=1.2,
    )

    _save(fig, "olmo_1b_0724_pipeline")


def make_low_rank_figure() -> None:
    fig = plt.figure(figsize=(14.4, 6.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fbf8f3")

    ax.text(
        0.5,
        0.945,
        "Low-Rank and PHLoRA Expert Rewrites as Declarative Checkpoint Surgery",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.902,
        "A validated MoE checkpoint becomes the common starting point for two complementary expert-compression workflows.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    boxes = [
        (0.05, 0.37, 0.18, 0.22, "#e8eef9", "Validated\n2-expert MoE\ncheckpoint"),
        (0.31, 0.52, 0.22, 0.20, "#dff3f1", "BrainSurgery YAML:\nPHLoRA factorization\nof expert-1 deltas"),
        (0.31, 0.16, 0.22, 0.20, "#dff3f1", "BrainSurgery YAML:\nlow-rank in-place\nexpert rewrite"),
        (0.61, 0.52, 0.16, 0.20, "#eef7e8", "FlexMoRE-style\nPHLoRA output"),
        (0.61, 0.16, 0.16, 0.20, "#eef7e8", "Dense MoE with\nlow-rank expert 1"),
        (0.82, 0.52, 0.13, 0.20, "#edf7ed", "Reference\nPython\n+ diff"),
        (0.82, 0.16, 0.13, 0.20, "#edf7ed", "Reference\nPython\n+ diff"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#1f2937",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            family="DejaVu Serif",
            color="#17202a",
            fontweight="bold",
            linespacing=1.15,
        )

    arrows = [
        ((0.23, 0.52), (0.31, 0.62)),
        ((0.23, 0.45), (0.31, 0.26)),
        ((0.53, 0.62), (0.61, 0.62)),
        ((0.53, 0.26), (0.61, 0.26)),
        ((0.77, 0.62), (0.82, 0.62)),
        ((0.77, 0.26), (0.82, 0.26)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#7c3f00")
        )

    _callout(
        ax,
        0.305,
        0.75,
        0.235,
        0.075,
        "Expert 1 becomes explicit PHLoRA factors\nrelative to dense expert 0.",
    )
    _callout(
        ax,
        0.305,
        0.055,
        0.235,
        0.075,
        "Expert 1 stays a standard dense tensor\nbut is rewritten by a rank-limited delta.",
    )

    ax.text(
        0.5,
        0.025,
        "Both branches stay in the same reproducible pattern: YAML surgery -> reference implementation -> diff-based validation.",
        ha="center",
        va="center",
        fontsize=10.5,
        family="DejaVu Serif",
        color="#334155",
    )

    _save(fig, "olmo_1b_0724_low_rank")


def make_axon_synapse_figure() -> None:
    fig = plt.figure(figsize=(14.2, 5.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.5,
        0.94,
        "How BrainSurgery Fits with Axon and Synapse",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.895,
        "BrainSurgery edits checkpoint weights; Axon and Synapse describe executable model structure.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    boxes = [
        (0.05, 0.52, 0.2, 0.18, "#f7efe8", "Axon DSL\nhuman-authored\nmodel graph"),
        (0.36, 0.52, 0.2, 0.18, "#e8eef9", "Synapse YAML\nstructured declarative\nmodel spec"),
        (0.67, 0.52, 0.2, 0.18, "#eef7e8", "Generated / runtime\nPyTorch model"),
        (0.05, 0.17, 0.2, 0.18, "#dff3f1", "BrainSurgery YAML\ncheckpoint surgery\nplans"),
        (0.36, 0.17, 0.2, 0.18, "#edf7ed", "Converted / validated\ncheckpoint artifacts"),
        (0.67, 0.17, 0.2, 0.18, "#f3e3cf", "Bridge example:\nAxon graph aligned to\nrewritten checkpoint"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=1.4,
                edgecolor="#1f2937",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=12,
            family="DejaVu Serif",
            color="#17202a",
            fontweight="bold",
            linespacing=1.18,
        )

    arrows = [
        ((0.25, 0.61), (0.36, 0.61)),
        ((0.56, 0.61), (0.67, 0.61)),
        ((0.25, 0.26), (0.36, 0.26)),
        ((0.56, 0.26), (0.67, 0.26)),
        ((0.46, 0.35), (0.46, 0.52)),
        ((0.77, 0.35), (0.77, 0.52)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, lw=2.0, color="#7c3f00")
        )

    _callout(
        ax,
        0.31,
        0.77,
        0.30,
        0.07,
        "Axon is the readable authoring language; Synapse is the structured model specification.",
    )
    _callout(
        ax,
        0.31,
        0.04,
        0.30,
        0.07,
        "Checkpoint surgery and executable model structure stay separate, but connect cleanly.",
    )

    _save(fig, "olmo_1b_0724_axon_synapse")


def make_low_rank_comparison_figure() -> None:
    fig = plt.figure(figsize=(15.5, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f6f1e8")

    ax.text(
        0.5,
        0.965,
        "Low-Rank Expert Rewriting: Imperative Reference vs Declarative BrainSurgery",
        ha="center",
        va="top",
        fontsize=19,
        fontweight="bold",
        family="DejaVu Serif",
        color="#17202a",
    )
    ax.text(
        0.5,
        0.93,
        "The same declarative pattern extends from structural MoE upcycling to representation-level expert compression.",
        ha="center",
        va="top",
        fontsize=11,
        family="DejaVu Serif",
        color="#334155",
    )

    _panel(ax, 0.035, 0.13, 0.28, 0.73, "A. Reference Python Script", "#9f2f21")
    _panel(ax, 0.36, 0.13, 0.28, 0.73, "B. BrainSurgery YAML Plan", "#155e63")
    _panel(ax, 0.685, 0.13, 0.28, 0.73, "C. Built-In Validation", "#2f5f2f")

    _code_block(
        ax,
        0.055,
        0.755,
        [
            "for layer in range(16):",
            '  for proj in ("gate_proj", "up_proj", "down_proj"):',
            '    expert0_key = f"model.layers.{layer}.mlp..."',
            '    expert1_key = f"model.layers.{layer}.mlp..."',
            "    delta = source[expert1_key] - source[expert0_key]",
            "    approx_delta = reconstruct_phlora_rank(",
            "      delta,",
            "      rank,",
            "      cache=cache,",
            "      cache_key=expert1_key,",
            "      ...",
            "    )",
            "    out[expert1_key] =",
            "      source[expert0_key] + approx_delta",
        ],
    )
    _callout(
        ax,
        0.05,
        0.145,
        0.25,
        0.07,
        "Low-rank approximation logic is handwritten\ninside loops and tensor mutation code.",
    )

    _code_block(
        ax,
        0.38,
        0.755,
        [
            "transforms:",
            "  - copy: { from: expert_1, to: expert_1.delta }",
            "  - subtract_:",
            "      from: model::...experts.0...",
            "      to:   model::...experts.1...delta...",
            "  - phlora:",
            "      target:   model::...experts.1...delta...",
            "      target_a: model::...phlora_a.weight",
            "      target_b: model::...phlora_b.weight",
            "      rank: 64",
            "  - delete: { target: model::...experts.1.weight }",
            "  - assert:",
            "      shape: { of: model::...phlora_a.weight, is: [64, ...] }",
            "  - assert:",
            "      not: { exists: model::...experts.1.weight }",
        ],
    )
    _callout(
        ax,
        0.375,
        0.145,
        0.25,
        0.07,
        "Compression, cleanup, and safety checks are\nexpressed directly in the checkpoint plan.",
    )

    _code_block(
        ax,
        0.705,
        0.755,
        [
            "inputs:",
            "  - yaml::olmo_1b_0724_hf_low_rank_expert_r64_demo",
            "  - ref::olmo_1b_0724_hf_low_rank_expert_r64_reference",
            "",
            "transforms:",
            "  - diff: { mode: aliases, left_alias: ref,",
            "            right_alias: yaml }",
            "",
            "Missing on left:",
            "  (none)",
            "Missing on right:",
            "  (none)",
            "Differing:",
            "  (none)",
            "No differences found.",
        ],
    )
    _callout(
        ax,
        0.7,
        0.145,
        0.25,
        0.07,
        "The low-rank rewrite is also validated\nagainst an independent reference.",
    )

    ax.add_patch(
        FancyArrowPatch((0.315, 0.495), (0.36, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )
    ax.add_patch(
        FancyArrowPatch((0.64, 0.495), (0.685, 0.495), arrowstyle="-|>", mutation_scale=18, lw=2.2, color="#7c3f00")
    )

    ax.text(
        0.5,
        0.07,
        "Validated MoE checkpoint  ->  low-rank expert rewrite  ->  reference implementation  ->  BrainSurgery diff",
        ha="center",
        va="center",
        fontsize=12,
        family="DejaVu Serif",
        color="#17202a",
    )

    _save(fig, "olmo_1b_0724_low_rank_comparison")


def main() -> None:
    make_comparison_figure()
    make_pipeline_figure()
    make_low_rank_figure()
    make_axon_synapse_figure()
    make_low_rank_comparison_figure()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
