library(ggplot2)

setwd("F:/scientific_research/aging_project/all_count/20260129/aging_clock/intergration5/all_coefficients/")

# ========== 可选设置 ==========
use_log_expression <- FALSE   # TRUE = 表达量取 log2 后再做 PCA；FALSE = 使用原始 TMM+CPM
output_dir <- "PCA_plots"     # PCA 图保存文件夹
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

# 读取基因列表文件：
# 该文件包含多个列，每一列（如 Elastic_Net、LASSO、MoE、Random_Forest、SVR、XGBoost）
# 对应一组基因，我们将对每一列的基因分别做 PCA，并最终拼成 3x2 的大图。
gene_list_df <- read.table("coefficients_HPO_all_top100.txt", header = TRUE, sep = "\t", check.names = FALSE)

# 若表达矩阵列名带前缀如 "4605:GCNT4"，用冒号后部分匹配
col_names <- colnames(t_data)
match_gene_col <- function(g) {
  g <- trimws(g)
  if (g %in% col_names) return(g)
  suffix_match <- which(sapply(strsplit(col_names, ":"), function(x) tail(x, 1)) == g)
  if (length(suffix_match) > 0) return(col_names[suffix_match[1]])
  return(NULL)
}

# 辅助函数：根据一列基因名做 PCA，返回 list(plot = ggplot对象, metrics = 统计指标)
run_pca_for_gene_vector <- function(genes, method_name) {
  genes <- as.character(genes)
  genes <- genes[!is.na(genes) & nchar(trimws(genes)) > 0]

  # 得到 (基因名, 表达矩阵列名) 对应，去重
  gene_col_map <- character(0)
  seen_cols <- character(0)
  for (g in genes) {
    col <- match_gene_col(g)
    if (!is.null(col) && !col %in% seen_cols) {
      gene_col_map[trimws(g)] <- col
      seen_cols <- c(seen_cols, col)
    }
  }

  if (length(gene_col_map) < 2) {
    warning("方法 ", method_name, " 用于 PCA 的基因数少于 2，跳过该方法。")
    return(NULL)
  }

  message("方法 ", method_name, " 共匹配到 ", length(gene_col_map), " 个基因用于 PCA。")

  # 以该方法的基因为特征子集构建表达矩阵
  expr_mat <- t_data[, unname(gene_col_map), drop = FALSE]

  if (!"Chronological_Age" %in% colnames(info)) {
    stop("metadata 中未找到列 'Chronological_Age'，请检查 total_metadata.txt。")
  }

  # 按样本名强制对齐表达矩阵与 metadata，避免顺序不一致导致年龄错配
  common_samples <- intersect(rownames(expr_mat), rownames(info))
  if (length(common_samples) == 0) {
    stop("表达矩阵与 metadata 中没有共同的样本名，请检查两个文件的样本命名是否一致。")
  }
  if (length(common_samples) < nrow(expr_mat)) {
    warning("部分样本在 metadata 中未找到，已自动过滤，剩余 ", length(common_samples), " 个样本。")
  }
  expr_mat <- expr_mat[common_samples, , drop = FALSE]
  age_vec  <- info[common_samples, "Chronological_Age"]

  # PCA（按基因维度中心化并标准化）
  set.seed(123)
  pca_res <- prcomp(expr_mat, center = TRUE, scale. = TRUE)

  # 取前两个主成分，并与年龄信息合并
  expl_var <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
  pc1_var <- round(expl_var[1] * 100, 1)
  pc2_var <- round(expl_var[2] * 100, 1)

  pca_df <- data.frame(
    Sample = rownames(expr_mat),
    PC1 = pca_res$x[, 1],
    PC2 = pca_res$x[, 2],
    Age = age_vec
  )

  # ===== 年龄梯度定量评估 =====
  # 1. PC1 与年龄的 Spearman 相关系数（及 p 值）
  cor_test_pc1 <- cor.test(pca_df$PC1, pca_df$Age, method = "spearman")
  rho_pc1    <- round(cor_test_pc1$estimate, 3)
  pval_pc1   <- cor_test_pc1$p.value

  # 2. PC2 与年龄的 Spearman 相关系数
  cor_test_pc2 <- cor.test(pca_df$PC2, pca_df$Age, method = "spearman")
  rho_pc2    <- round(cor_test_pc2$estimate, 3)
  pval_pc2   <- cor_test_pc2$p.value

  # 3. 线性回归 Age ~ PC1 + PC2 的 R²（反映前两个主成分联合解释年龄方差的能力）
  lm_fit  <- lm(Age ~ PC1 + PC2, data = pca_df)
  r2_pc12 <- round(summary(lm_fit)$r.squared, 3)
  pval_lm <- round(summary(lm_fit)$fstatistic, 3)
  # F 检验 p 值
  fstat   <- summary(lm_fit)$fstatistic
  pval_lm_f <- pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)

  metrics <- data.frame(
    Method          = method_name,
    N_genes         = length(gene_col_map),
    N_samples       = nrow(pca_df),
    PC1_var_pct     = pc1_var,
    PC2_var_pct     = pc2_var,
    Spearman_rho_PC1 = rho_pc1,
    Spearman_p_PC1   = signif(pval_pc1, 3),
    Spearman_rho_PC2 = rho_pc2,
    Spearman_p_PC2   = signif(pval_pc2, 3),
    R2_Age_PC1_PC2  = r2_pc12,
    Pval_lm_F       = signif(pval_lm_f, 3),
    stringsAsFactors = FALSE
  )

  message(sprintf(
    "  [%s] Spearman(PC1,Age)=%.3f (p=%.2e) | R²(Age~PC1+PC2)=%.3f",
    method_name, rho_pc1, pval_pc1, r2_pc12
  ))

  # 在图上添加统计注释（左上角）
  annot_label <- paste0(
    "rho(PC1)=", rho_pc1,
    "\nR\u00b2(PC1+PC2)=", r2_pc12
  )

  # 绘制 PCA 散点图：PC1 vs PC2，按年龄渐变着色
  p <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Age)) +
    geom_point(size = 3.5, alpha = 1) +
    scale_color_gradient(
      low = "#fdd0a2",
      high = "#a63603",
      name = "Age"
    ) +
    labs(
      title = method_name,
      x = paste0("PC1 (", pc1_var, "% var)"),
      y = paste0("PC2 (", pc2_var, "% var)")
    ) +
    annotate(
      "text",
      x = -Inf, y = Inf,
      hjust = -0.05, vjust = 1.3,
      label = annot_label,
      size = 4, color = "black"
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

  return(list(plot = p, metrics = metrics))
}

# 创建输出文件夹
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
message("PCA 图将保存到文件夹: ", output_dir)

# 对每一列方法依次做 PCA 并收集图形
method_names <- colnames(gene_list_df)
plot_list <- list()

for (m in method_names) {
  message("开始处理方法: ", m)
  genes_m <- gene_list_df[[m]]
  p_m <- run_pca_for_gene_vector(genes_m, m)
  if (!is.null(p_m)) {
    plot_list[[m]] <- p_m

    # 单独保存每个方法的图
    out_file_png <- file.path(output_dir, paste0("PCA_", m, ".png"))
    # 每个单独的 PCA 图稍微加宽一些（原来 width = 7）
    ggsave(out_file_png, plot = p_m, width = 8, height = 5, dpi = 150, bg = "white")
    message("已保存 PCA 图: ", out_file_png)

    out_file_pdf <- file.path(output_dir, paste0("PCA_", m, ".pdf"))
    ggsave(out_file_pdf, plot = p_m, width = 8, height = 5)
    message("已保存 PCA PDF 图: ", out_file_pdf)
  }
}

if (length(plot_list) == 0) {
  stop("所有方法的基因数都不足以进行 PCA（<2 个基因），未生成任何图形。")
}

# 使用 patchwork 或 gridExtra 合并图形为 3x2 大图
if (!requireNamespace("patchwork", quietly = TRUE)) {
  message("未检测到 patchwork 包，将使用 gridExtra 进行拼图。")
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    stop("需要安装 'patchwork' 或 'gridExtra' 包以拼接 3x2 大图。")
  } else {
    # gridExtra 版本
    gridExtra::grid.arrange(grobs = plot_list, ncol = 3, nrow = 2)
    out_file_big_png <- file.path(output_dir, "PCA_all_methods_3x2.png")
    # 合并图整体也适当加宽
    ggsave(out_file_big_png, width = 17, height = 10, dpi = 150, bg = "white")

    out_file_big_pdf <- file.path(output_dir, "PCA_all_methods_3x2.pdf")
    ggsave(out_file_big_pdf, width = 17, height = 10)
  }
} else {
  # patchwork 版本
  library(patchwork)
  # 将列表顺序固定
  plts <- plot_list
  # 如果刚好是 6 个图，按 3x2 排列；否则按照行列数自动排
  combined <- NULL
  n_plots <- length(plts)
  if (n_plots >= 6) {
    combined <- (plts[[1]] | plts[[2]] | plts[[3]]) /
      (plts[[4]] | plts[[5]] | plts[[6]])
  } else if (n_plots > 1) {
    # 少于 6 个图时，简单按一行排布
    combined <- Reduce(`|`, plts)
  } else {
    combined <- plts[[1]]
  }

  out_file_big_png <- file.path(output_dir, "PCA_all_methods_3x2.png")
  # 合并图整体也适当加宽
  ggsave(out_file_big_png, plot = combined, width = 17, height = 10, dpi = 150, bg = "white")
  message("已保存 3x2 合并 PCA 图: ", out_file_big_png)

  out_file_big_pdf <- file.path(output_dir, "PCA_all_methods_3x2.pdf")
  ggsave(out_file_big_pdf, plot = combined, width = 17, height = 10)
  message("已保存 3x2 合并 PCA PDF 图: ", out_file_big_pdf)
}