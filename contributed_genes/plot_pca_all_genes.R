library(ggplot2)

setwd("F:/scientific_research/aging_project/all_count/20260129/aging_clock/intergration5/all_coefficients/")

# ========== 可选设置 ==========
use_log_expression <- FALSE      # TRUE = 表达量取 log2 后再做 PCA；FALSE = 使用原始 TMM+CPM
output_dir <- "PCA_all_genes"    # 输出文件夹
plot_width <- 17 / 3             # 与原 3x2 总图中单个小图宽度一致
plot_height <- 10 / 2            # 与原 3x2 总图中单个小图高度一致
# ==============================

# 读取表达矩阵与样本信息
data <- read.table("total_count_sort_ortholog_TMM_batched.txt", header = TRUE, row.names = 1, check.names = FALSE)
info <- read.table("total_metadata.txt", header = TRUE, row.names = 1, check.names = FALSE)

# 表达矩阵转置：行为样本，列为基因
t_data <- as.data.frame(t(data))

# 可选：表达量取对数（log2），若有 0 或负值则先平移
if (use_log_expression) {
  min_val <- min(as.matrix(t_data), na.rm = TRUE)
  offset <- if (min_val < 1) 1 - min_val else 0
  t_data <- as.data.frame(log2(as.matrix(t_data) + offset))
  message("表达量已转换为 log2(value + ", offset, ")")
}

if (!"Chronological_Age" %in% colnames(info)) {
  stop("metadata 中未找到列 'Chronological_Age'，请检查 total_metadata.txt。")
}

# 按样本名强制对齐表达矩阵与 metadata
common_samples <- intersect(rownames(t_data), rownames(info))
if (length(common_samples) == 0) {
  stop("表达矩阵与 metadata 中没有共同的样本名，请检查两个文件的样本命名是否一致。")
}
if (length(common_samples) < nrow(t_data)) {
  warning("部分样本在 metadata 中未找到，已自动过滤，剩余 ", length(common_samples), " 个样本。")
}

expr_mat <- t_data[common_samples, , drop = FALSE]
age_vec <- info[common_samples, "Chronological_Age"]

# 去除方差为 0 或缺失的基因，避免 prcomp(scale.=TRUE) 报错
gene_var <- apply(expr_mat, 2, var, na.rm = TRUE)
keep_genes <- !is.na(gene_var) & gene_var > 0
expr_mat <- expr_mat[, keep_genes, drop = FALSE]

if (ncol(expr_mat) < 2) {
  stop("过滤后可用于 PCA 的基因数少于 2，无法进行 PCA。")
}

message("用于 PCA 的样本数: ", nrow(expr_mat))
message("用于 PCA 的基因数: ", ncol(expr_mat))

# 创建输出文件夹
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
message("输出文件夹: ", output_dir)

# PCA（按基因维度中心化并标准化）
set.seed(123)
pca_res <- prcomp(expr_mat, center = TRUE, scale. = TRUE)

# 解释方差比例
expl_var <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
pc1_var <- round(expl_var[1] * 100, 1)
pc2_var <- round(expl_var[2] * 100, 1)
pc1_var_label <- sprintf("%.1f", pc1_var)
pc2_var_label <- sprintf("%.1f", pc2_var)

# 主成分坐标
pca_df <- data.frame(
  Sample = rownames(expr_mat),
  PC1 = pca_res$x[, 1],
  PC2 = pca_res$x[, 2],
  Age = age_vec,
  stringsAsFactors = FALSE
)

# 年龄梯度定量评估
lm_fit <- lm(Age ~ PC1 + PC2, data = pca_df)
r2_pc12 <- round(summary(lm_fit)$r.squared, 3)
r2_pc12_label <- sprintf("%.3f", r2_pc12)
fstat <- summary(lm_fit)$fstatistic
pval_lm_f <- pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)

metrics_df <- data.frame(
  Analysis = "All_Genes",
  N_genes = ncol(expr_mat),
  N_samples = nrow(pca_df),
  PC1_var_pct = pc1_var,
  PC2_var_pct = pc2_var,
  R2_Age_PC1_PC2 = r2_pc12,
  Pval_lm_F = signif(pval_lm_f, 3),
  stringsAsFactors = FALSE
)

message(sprintf(
  "[All_Genes] R²(Age~PC1+PC2)=%.3f (F-test p=%.2e)",
  r2_pc12, pval_lm_f
))

# PCA 散点图
annot_label <- paste0("R\u00b2(PC1+PC2)=", r2_pc12_label)

p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Age)) +
  geom_point(size = 3.5, alpha = 1) +
  scale_color_gradient(
    low = "#fdd0a2",
    high = "#a63603",
    name = "Age"
  ) +
  labs(
    title = "PCA of All Genes",
    x = paste0("PC1 (", pc1_var_label, "% var)"),
    y = paste0("PC2 (", pc2_var_label, "% var)")
  ) +
  annotate(
    "text",
    x = Inf, y = -Inf,
    hjust = 1.05, vjust = -0.5,
    label = annot_label,
    size = 5.5, color = "black", fontface = "bold"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12),
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    axis.line = element_line(color = "black", linewidth = 0.5)
  )

# 解释方差柱状图（前 10 个主成分）
pc_bar_df <- data.frame(
  PC = factor(paste0("PC", seq_len(min(10, length(expl_var)))),
              levels = paste0("PC", seq_len(min(10, length(expl_var))))),
  Variance = round(expl_var[seq_len(min(10, length(expl_var)))] * 100, 1),
  stringsAsFactors = FALSE
)

p_var <- ggplot(pc_bar_df, aes(x = PC, y = Variance)) +
  geom_bar(stat = "identity", width = 0.7, fill = "#d6604d") +
  labs(
    title = "Explained Variance of Top PCs",
    x = "Principal Component",
    y = "Explained Variance (%)"
  ) +
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# 输出结果
write.table(
  pca_df,
  file.path(output_dir, "PCA_all_genes_coordinates.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

write.table(
  metrics_df,
  file.path(output_dir, "PCA_all_genes_metrics.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

write.table(
  data.frame(
    PC = paste0("PC", seq_along(expl_var)),
    Explained_Variance_Ratio = expl_var,
    Explained_Variance_Percent = expl_var * 100,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "PCA_all_genes_explained_variance.txt"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

ggsave(file.path(output_dir, "PCA_all_genes.png"), plot = p, width = plot_width, height = plot_height, dpi = 150, bg = "white")
ggsave(file.path(output_dir, "PCA_all_genes.pdf"), plot = p, width = plot_width, height = plot_height)

ggsave(file.path(output_dir, "PCA_all_genes_variance_barplot.png"), plot = p_var, width = plot_width, height = plot_height, dpi = 150, bg = "white")
ggsave(file.path(output_dir, "PCA_all_genes_variance_barplot.pdf"), plot = p_var, width = plot_width, height = plot_height)

message("已保存 PCA 散点图、解释方差柱状图、坐标表和统计表。")
