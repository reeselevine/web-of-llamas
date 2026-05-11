import matplotlib.pyplot as plt
import numpy as np

# Raw ggml_webgpu profiling percentages by workload/kernel, llama3.2 1b q4_k_m.
# Values are percentages of total GPU shader time.
raw = {
    "pp512": {
        "RMS_NORM_MUL_inplace": 0.00,
        "mul_mat_vec_q6_K_f32_sg_reduce": 0.41,
        "set_rows_dstf16_vec4_i64idx": 0.07,
        "rope_f32_ff": 0.24,
        "mul_mat_vec_q4_K_f32_sg_reduce": 0.04,
        "get_rows_f32_vec": 0.00,
        "glu_swiglu_f32_split": 1.00,
        "mul_mat_subgroup_matrix_q6_K_f32": 14.53,
        "flash_attn_f16_mask_kvdirect_hsqk64_hsv64": 4.44,
        "ADD_f32_inplace": 0.67,
        "mul_mat_subgroup_matrix_q4_K_f32": 78.06,
        "cpy_f32_f16": 0.01,
        "RMS_NORM_MUL": 0.53,
    },
    "tg128": {
        "RMS_NORM_MUL_inplace": 0.04,
        "get_rows_f32_vec": 0.05,
        "glu_swiglu_f32_split": 0.52,
        "flash_attn_vec_blk_kvt32_wg1024": 2.98,
        "ADD_f32_inplace": 1.06,
        "cpy_f32_f16": 0.03,
        "RMS_NORM_MUL": 1.42,
        "mul_mat_vec_q4_K_f32_sg_reduce": 57.32,
        "set_rows_dstf16_vec4_i64idx": 0.81,
        "rope_f32_ff": 0.87,
        "mul_mat_vec_q6_K_f32_sg_reduce": 32.58,
        "flash_attn_vec_reduce_hsv64_wg128": 0.50,
        "flash_attn_vec_f16_mask_blk_kvdirect_hsqk64_hsv64": 1.83,
    },
    "pp512 @ d2048": {
        "RMS_NORM_MUL_inplace": 0.00,
        "mul_mat_vec_q6_K_f32_sg_reduce": 0.26,
        "set_rows_dstf16_vec4_i64idx": 0.06,
        "rope_f32_ff": 0.20,
        "mul_mat_vec_q4_K_f32_sg_reduce": 0.03,
        "get_rows_f32_vec": 0.00,
        "glu_swiglu_f32_split": 0.90,
        "mul_mat_subgroup_matrix_q6_K_f32": 13.11,
        "flash_attn_f16_mask_kvdirect_hsqk64_hsv64": 14.06,
        "ADD_f32_inplace": 0.60,
        "mul_mat_subgroup_matrix_q4_K_f32": 70.28,
        "cpy_f32_f16": 0.02,
        "RMS_NORM_MUL": 0.47,
    },
    "tg128 @ d2048": {
        "flash_attn_vec_reduce_hsv64_wg1024": 0.55,
        "RMS_NORM_MUL_inplace": 0.04,
        "get_rows_f32_vec": 0.05,
        "glu_swiglu_f32_split": 0.44,
        "flash_attn_vec_blk_kvt32_wg1024": 5.21,
        "ADD_f32_inplace": 0.96,
        "cpy_f32_f16": 0.03,
        "RMS_NORM_MUL": 1.29,
        "mul_mat_vec_q4_K_f32_sg_reduce": 51.50,
        "set_rows_dstf16_vec4_i64idx": 0.73,
        "rope_f32_ff": 0.78,
        "mul_mat_vec_q6_K_f32_sg_reduce": 29.24,
        "flash_attn_vec_reduce_hsv64_wg128": 0.00,
        "flash_attn_vec_f16_mask_blk_kvdirect_hsqk64_hsv64": 9.20,
    },
}

# Explicit kernel -> category grouping.
groups = {
    "Quantized MatMul": [
        "mul_mat_subgroup_matrix_q4_K_f32",
        "mul_mat_subgroup_matrix_q6_K_f32",
    ],
    "Quantized MatVec": [
        "mul_mat_vec_q4_K_f32_sg_reduce",
        "mul_mat_vec_q6_K_f32_sg_reduce",
    ],
    "Attention": [
        "flash_attn_f16_mask_kvdirect_hsqk64_hsv64",
        "flash_attn_vec_blk_kvt32_wg1024",
        "flash_attn_vec_reduce_hsv64_wg1024",
        "flash_attn_vec_reduce_hsv64_wg128",
        "flash_attn_vec_f16_mask_blk_kvdirect_hsqk64_hsv64",
    ],
    "Norm / Elementwise": [
        "RMS_NORM_MUL_inplace",
        "RMS_NORM_MUL",
        "ADD_f32_inplace",
        "glu_swiglu_f32_split",
    ],
    "Other": [
        "set_rows_dstf16_vec4_i64idx",
        "rope_f32_ff",
        "get_rows_f32_vec",
        "cpy_f32_f16",
    ],
}

workloads = list(raw.keys())

# Aggregate raw kernel percentages into explicit groups.
grouped = {
    group: [
        sum(raw[w].get(kernel, 0.0) for kernel in kernels)
        for w in workloads
    ]
    for group, kernels in groups.items()
}

# Normalize each workload to 100%, since rounded percentages may not sum exactly.
totals = np.array([
    sum(grouped[group][i] for group in groups)
    for i in range(len(workloads))
])

grouped_norm = {
    group: 100 * np.array(values) / totals
    for group, values in grouped.items()
}

print("Grouped percentages:")
for w_idx, workload in enumerate(workloads):
    print(f"\n{workload}")
    for group in groups:
        print(f"  {group:20s}: {grouped_norm[group][w_idx]:6.2f}%")

column_labels = {
    "pp512": "Prefill d1024",
    "tg128": "Decode d1024",
    "pp512 @ d2048": "Prefill d2048",
    "tg128 @ d2048": "Decode d2048",
}

print("\nLaTeX table:")
print(r"\begin{table}[t]")
print(r"\footnotesize")
print(r"\caption{GPU shader time breakdown by workload category.}")
print(r"\label{tab:webgpu-kernel-breakdown}")
print(r"\centering")
print(r"\begin{tabular}{l r r r r}")
print(r"\toprule")
print(
    r"\textbf{Category} & "
    + " & ".join(
        rf"\textbf{{{column_labels[workload]}}}"
        for workload in workloads
    )
    + r" \\"
)
print(r"\midrule")
for group in groups:
    values = " & ".join(
        f"{grouped_norm[group][w_idx]:.2f}\\%"
        for w_idx in range(len(workloads))
    )
    print(f"{group} & {values} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

x = np.arange(len(workloads))
bar_width = 0.16

colors = {
    "Quantized MatMul": "#7da9d8",
    "Quantized MatVec": "#f2b37e",
    "Attention": "#8cc89a",
    "Norm / Elementwise": "#d8a6cf",
    "Other": "#b39ddb",
}

fig, ax = plt.subplots(figsize=(7.0, 3.2))
log_floor = 0.1

seen_labels = set()
for w_idx, workload in enumerate(workloads):
    present_groups = [
        group for group in groups
        if grouped_norm[group][w_idx] > 0
    ]
    offsets = (
        np.arange(len(present_groups)) - (len(present_groups) - 1) / 2
    ) * bar_width

    for offset, group in zip(offsets, present_groups):
        label = group if group not in seen_labels else None
        value = grouped_norm[group][w_idx]
        ax.bar(
            x[w_idx] + offset,
            value - log_floor,
            width=bar_width,
            bottom=log_floor,
            label=label,
            color=colors[group],
        )
        seen_labels.add(group)

ax.set_ylabel("GPU time breakdown (%), log scale")
ax.set_yscale("log")
ax.set_ylim(log_floor, 100)
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.22),
    ncol=3,
    frameon=False,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig("webgpu_kernel_breakdown.pdf", bbox_inches="tight")

plt.show()
