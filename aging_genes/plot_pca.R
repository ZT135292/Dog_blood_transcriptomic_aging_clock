library(ggplot2)

script_dir <- tryCatch(
  dirname(normalizePath(sys.frame(1)$ofile, winslash = "/", mustWork = TRUE)),
  error = function(e) getwd()
)
setwd(script_dir)

counts_file <- "../total_count_sort_ortholog.txt"
metadata_file <- "../total_metadata.txt"
gene_list_file <- "coefficients_HPO_all_top100.txt"
normal_condition_label <- "Normal"

filter_low_expression <- TRUE
prior_count <- 1
save_logcpm_matrix <- TRUE
logcpm_output_file <- "total_count_sort_ortholog_TMM_logcpm_normal_ref.txt"

enable_combat <- TRUE
combat_age_col <- "Chronological_Age"
logcpm_combat_output_file <- "total_count_sort_ortholog_TMM_logcpm_normal_ref_combat.txt"

pca_samples <- "normal"
use_log_expression <- FALSE
output_dir <- if (enable_combat) "PCA_plots_combat" else "PCA_plots"
metrics_output_file <- "PCA_age_gradient_metrics.txt"

enable_batch_analysis <- TRUE
batch_column <- "batch"

enable_all_genes_pca <- TRUE
all_genes_method_label <- "All_genes"

enable_random_100_genes_pca <- TRUE
random_100_genes_n <- 100
random_100_genes_replicates <- 5
random_100_genes_seed <- 66
random_100_genes_method_prefix <- "Random100"

if (!requireNamespace("edgeR", quietly = TRUE)) {
  stop("需要安装 edgeR：在 R 中运行 BiocManager::install('edgeR')")
}

counts <- read.table(
  counts_file, header = TRUE, row.names = 1, sep = "\t", check.names = FALSE
)
info <- read.table(
  metadata_file, header = TRUE, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE
)

if (!"sample_id" %in% colnames(info)) {
  stop("metadata 中未找到列 'sample_id'，请检查 ", metadata_file)
}
if (!"Chronological_Age" %in% colnames(info)) {
  stop("metadata 中未找到列 'Chronological_Age'，请检查 ", metadata_file)
}
if (!"condition" %in% colnames(info)) {
  stop("metadata 中未找到列 'condition'，模式 B 需要据此识别 Normal 参考样本。")
}
if (enable_batch_analysis && !batch_column %in% colnames(info)) {
  warning("未找到 batch 列 '", batch_column, "'，已关闭 batch 分析。")
  enable_batch_analysis <- FALSE
}

pca_theme <- function() {
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
}

common_samples <- intersect(colnames(counts), info$sample_id)
if (length(common_samples) < 2) {
  stop("counts 与 metadata 无足够共同样本，请检查 sample_id 是否一致。")
}
if (length(common_samples) < ncol(counts) || length(common_samples) < nrow(info)) {
  message(
    "仅使用 counts 与 metadata 交集样本: ",
    length(common_samples), " / counts ", ncol(counts), ", metadata ", nrow(info)
  )
}

counts <- counts[, common_samples, drop = FALSE]
info <- info[match(common_samples, info$sample_id), , drop = FALSE]

normal_samples <- info$sample_id[info$condition == normal_condition_label]
normal_samples <- intersect(normal_samples, common_samples)
if (length(normal_samples) < 2) {
  stop("Normal 参考样本不足 2 个，无法估计 TMM。请检查 condition 列与 ", normal_condition_label)
}

non_normal_samples <- setdiff(common_samples, normal_samples)
message(
  "TMM 参考: ", length(normal_samples), " 个 Normal 样本; ",
  "另 ", length(non_normal_samples), " 个非 Normal 样本将沿用该参考计算 logCPM。"
)

normal_counts <- as.matrix(counts[, normal_samples, drop = FALSE])
storage.mode(normal_counts) <- "double"

if (filter_low_expression) {
  dge_norm <- edgeR::DGEList(counts = normal_counts)
  keep <- edgeR::filterByExpr(dge_norm)
  counts <- counts[rownames(dge_norm)[keep], , drop = FALSE]
  message("filterByExpr（仅 Normal）保留基因数: ", nrow(counts))
}

normal_counts <- as.matrix(counts[, normal_samples, drop = FALSE])
storage.mode(normal_counts) <- "double"

if (length(non_normal_samples) > 0) {
  other_counts <- as.matrix(counts[, non_normal_samples, drop = FALSE])
  storage.mode(other_counts) <- "double"
  combined_counts <- cbind(normal_counts, other_counts)
} else {
  combined_counts <- normal_counts
}

dge <- edgeR::DGEList(counts = combined_counts)
if (ncol(dge) >= 2) {
  dge <- edgeR::calcNormFactors(dge, method = "TMM", ref = 1)
} else {
  dge$samples$norm.factors <- 1
}

logcpm <- edgeR::cpm(
  dge,
  normalized.lib.sizes = TRUE,
  log = TRUE,
  prior.count = prior_count
)

if (save_logcpm_matrix) {
  write.table(
    logcpm,
    file = logcpm_output_file,
    sep = "\t",
    quote = FALSE,
    row.names = TRUE,
    col.names = NA
  )
  message("已保存 TMM+logCPM 矩阵（ComBat 前）: ", logcpm_output_file)
}

if (enable_combat) {
  if (!requireNamespace("sva", quietly = TRUE)) {
    stop("ComBat 需要 sva 包：BiocManager::install('sva')")
  }
  if (!combat_age_col %in% colnames(info)) {
    stop("ComBat 需要协变量列 '", combat_age_col, "'，请检查 metadata。")
  }
  if (!batch_column %in% colnames(info)) {
    stop("ComBat 需要 batch 列 '", batch_column, "'，请检查 metadata。")
  }

  combat_sample_ids <- colnames(logcpm)
  if (pca_samples == "normal") {
    combat_sample_ids <- intersect(combat_sample_ids, normal_samples)
  }
  if (length(combat_sample_ids) < 3) {
    stop("ComBat 样本数不足（< 3），请检查 pca_samples 与样本列表。")
  }

  combat_info <- info[match(combat_sample_ids, info$sample_id), , drop = FALSE]
  if (any(is.na(combat_info$sample_id))) {
    stop("ComBat 样本与 metadata 无法对齐。")
  }

  combat_mat <- as.matrix(logcpm[, combat_sample_ids, drop = FALSE])
  batch_vec <- factor(combat_info[[batch_column]])
  age_vec_combat <- combat_info[[combat_age_col]]
  mod <- model.matrix(~ age_vec_combat)
  colnames(mod)[2] <- combat_age_col

  n_batch_levels <- nlevels(batch_vec)
  if (n_batch_levels < 2) {
    stop("ComBat 需要至少 2 个 batch 水平，当前仅 ", n_batch_levels, " 个。")
  }

  message(
    "ComBat: ", length(combat_sample_ids), " 样本, ",
    n_batch_levels, " 个 batch, mod ~ ", combat_age_col
  )

  combat_mat_adj <- sva::ComBat(
    dat = combat_mat,
    batch = batch_vec,
    mod = mod,
    par.prior = TRUE,
    prior.plots = FALSE
  )
  logcpm[, combat_sample_ids] <- combat_mat_adj

  if (save_logcpm_matrix) {
    write.table(
      logcpm,
      file = logcpm_combat_output_file,
      sep = "\t",
      quote = FALSE,
      row.names = TRUE,
      col.names = NA
    )
    message("已保存 TMM+logCPM+ComBat 矩阵: ", logcpm_combat_output_file)
  }
}

t_data <- as.data.frame(t(logcpm))
rownames(t_data) <- colnames(logcpm)

if (pca_samples == "normal") {
  t_data <- t_data[normal_samples, , drop = FALSE]
  message("PCA 仅使用 Normal 样本: ", nrow(t_data))
} else if (pca_samples != "all") {
  stop("pca_samples 只能是 'all' 或 'normal'。")
} else {
  message("PCA 使用全部样本: ", nrow(t_data))
}

info <- info[match(rownames(t_data), info$sample_id), , drop = FALSE]
if (any(is.na(info$sample_id))) {
  stop("表达矩阵与 metadata 无法按 sample_id 完全对齐，请检查样本 ID。")
}

if (use_log_expression) {
  min_val <- min(as.matrix(t_data), na.rm = TRUE)
  offset <- if (min_val < 1) 1 - min_val else 0
  t_data <- as.data.frame(log2(as.matrix(t_data) + offset))
  message("表达量已额外转换为 log2(value + ", offset, ")")
}

gene_list_df <- read.table(
  gene_list_file, header = TRUE, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE
)

col_names <- colnames(t_data)
match_gene_col <- function(g) {
  g <- trimws(g)
  if (g %in% col_names) return(g)
  suffix_match <- which(sapply(strsplit(col_names, ":"), function(x) tail(x, 1)) == g)
  if (length(suffix_match) > 0) return(col_names[suffix_match[1]])
  return(NULL)
}

run_pca_for_gene_vector <- function(
  genes,
  method_name,
  info_df,
  do_batch_analysis = enable_batch_analysis,
  batch_col = batch_column
) {
  genes <- as.character(genes)
  genes <- genes[!is.na(genes) & nchar(trimws(genes)) > 0]

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

  expr_mat <- t_data[, unname(gene_col_map), drop = FALSE]

  common_samples <- intersect(rownames(expr_mat), info_df$sample_id)
  if (length(common_samples) == 0) {
    stop("表达矩阵与 metadata 中没有共同的样本名，请检查 sample_id 是否一致。")
  }
  if (length(common_samples) < nrow(expr_mat)) {
    warning(
      "方法 ", method_name, "：部分样本在 metadata 中未找到，已过滤，剩余 ",
      length(common_samples), " 个样本。"
    )
  }
  expr_mat <- expr_mat[common_samples, , drop = FALSE]
  matched_info <- info_df[match(common_samples, info_df$sample_id), , drop = FALSE]
  age_vec <- matched_info$Chronological_Age

  set.seed(123)
  pca_res <- prcomp(expr_mat, center = TRUE, scale. = TRUE)

  expl_var <- (pca_res$sdev^2) / sum(pca_res$sdev^2)
  pc1_var <- round(expl_var[1] * 100, 1)
  pc2_var <- round(expl_var[2] * 100, 1)

  pca_df <- data.frame(
    Sample = rownames(expr_mat),
    PC1 = pca_res$x[, 1],
    PC2 = pca_res$x[, 2],
    Age = age_vec,
    stringsAsFactors = FALSE
  )
  if (do_batch_analysis) {
    pca_df$Batch <- factor(matched_info[[batch_col]])
  }

  lm_fit <- lm(Age ~ PC1 + PC2, data = pca_df)
  r2_pc12 <- round(summary(lm_fit)$r.squared, 3)
  fstat <- summary(lm_fit)$fstatistic
  pval_lm_f <- pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)

  r2_pc12_adj_batch <- NA
  pval_lm_adj_f <- NA
  r2_batch_only <- NA
  r2_pc_incremental_after_batch <- NA
  if (do_batch_analysis) {
    lm_batch <- lm(Age ~ Batch, data = pca_df)
    r2_batch_only <- round(summary(lm_batch)$r.squared, 3)

    lm_adj <- lm(Age ~ PC1 + PC2 + Batch, data = pca_df)
    lm_adj_sum <- summary(lm_adj)
    r2_pc12_adj_batch <- round(lm_adj_sum$r.squared, 3)
    fstat_adj <- lm_adj_sum$fstatistic
    pval_lm_adj_f <- pf(fstat_adj[1], fstat_adj[2], fstat_adj[3], lower.tail = FALSE)

    r2_pc_incremental_after_batch <- round(r2_pc12_adj_batch - r2_batch_only, 3)
  }

  metrics <- data.frame(
    Method = method_name,
    N_genes = length(gene_col_map),
    N_samples = nrow(pca_df),
    PC1_var_pct = pc1_var,
    PC2_var_pct = pc2_var,
    R2_Age_PC1_PC2 = r2_pc12,
    Pval_lm_F = signif(pval_lm_f, 3),
    R2_Age_PC1_PC2_adj_batch = r2_pc12_adj_batch,
    Pval_lm_adj_batch_F = if (is.na(pval_lm_adj_f)) NA else signif(pval_lm_adj_f, 3),
    R2_Age_batch_only = r2_batch_only,
    R2_PC_incremental_after_batch = r2_pc_incremental_after_batch,
    stringsAsFactors = FALSE
  )

  log_msg <- sprintf(
    "  [%s] R²(Age~PC1+PC2)=%.3f (p=%.2e)",
    method_name, r2_pc12, pval_lm_f
  )
  if (do_batch_analysis) {
    log_msg <- paste0(
      log_msg,
      sprintf(
        " | R²(Age~PC1+PC2+batch)=%.3f | R²(batch)=%.3f | ΔR²(PC|batch)=%.3f",
        r2_pc12_adj_batch, r2_batch_only, r2_pc_incremental_after_batch
      )
    )
  }
  message(log_msg)

  annot_label <- paste0("R\u00b2(PC1+PC2)=", r2_pc12)
  if (do_batch_analysis) {
    annot_label <- paste0(
      annot_label,
      "\nR\u00b2(PC1+PC2+batch)=", r2_pc12_adj_batch
    )
  }

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
    pca_theme()

  p_batch <- NULL
  if (do_batch_analysis) {
    n_batch <- length(unique(pca_df$Batch))
    batch_palette <- if (n_batch <= 8) {
      RColorBrewer::brewer.pal(max(3, n_batch), "Set2")[seq_len(n_batch)]
    } else {
      grDevices::colorRampPalette(RColorBrewer::brewer.pal(8, "Set2"))(n_batch)
    }

    p_batch <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Batch)) +
      geom_point(size = 3.5, alpha = 1) +
      scale_color_manual(values = batch_palette, name = "Batch") +
      labs(
        title = paste0(method_name, " (by batch)"),
        x = paste0("PC1 (", pc1_var, "% var)"),
        y = paste0("PC2 (", pc2_var, "% var)")
      ) +
      pca_theme()
  }

  list(plot = p, plot_batch = p_batch, metrics = metrics)
}

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
message("PCA 图将保存到文件夹: ", output_dir)

method_names <- colnames(gene_list_df)
plot_list <- list()
plot_batch_list <- list()
metrics_list <- list()

if (enable_batch_analysis && !requireNamespace("RColorBrewer", quietly = TRUE)) {
  stop("batch 着色图需要 RColorBrewer 包：install.packages('RColorBrewer')")
}

for (m in method_names) {
  message("开始处理方法: ", m)
  genes_m <- gene_list_df[[m]]
  p_m <- run_pca_for_gene_vector(genes_m, m, info)
  if (!is.null(p_m)) {
    plot_list[[m]] <- p_m$plot
    metrics_list[[m]] <- p_m$metrics

    out_file_png <- file.path(output_dir, paste0("PCA_", m, ".png"))
    ggsave(out_file_png, plot = p_m$plot, width = 8, height = 5, dpi = 150, bg = "white")
    message("已保存 PCA 图: ", out_file_png)

    out_file_pdf <- file.path(output_dir, paste0("PCA_", m, ".pdf"))
    ggsave(out_file_pdf, plot = p_m$plot, width = 8, height = 5)
    message("已保存 PCA PDF 图: ", out_file_pdf)

    if (!is.null(p_m$plot_batch)) {
      plot_batch_list[[m]] <- p_m$plot_batch

      out_batch_png <- file.path(output_dir, paste0("PCA_", m, "_by_batch.png"))
      ggsave(out_batch_png, plot = p_m$plot_batch, width = 8, height = 5, dpi = 150, bg = "white")
      message("已保存 batch 着色 PCA 图: ", out_batch_png)

      out_batch_pdf <- file.path(output_dir, paste0("PCA_", m, "_by_batch.pdf"))
      ggsave(out_batch_pdf, plot = p_m$plot_batch, width = 8, height = 5)
      message("已保存 batch 着色 PCA PDF 图: ", out_batch_pdf)
    }
  }
}

if (enable_all_genes_pca) {
  message("开始全基因 PCA（", ncol(t_data), " 个基因）...")
  p_all <- run_pca_for_gene_vector(colnames(t_data), all_genes_method_label, info)
  if (!is.null(p_all)) {
    metrics_list[[all_genes_method_label]] <- p_all$metrics

    out_all_png <- file.path(output_dir, paste0("PCA_", all_genes_method_label, ".png"))
    ggsave(out_all_png, plot = p_all$plot, width = 8, height = 5, dpi = 150, bg = "white")
    message("已保存全基因 PCA 图: ", out_all_png)

    out_all_pdf <- file.path(output_dir, paste0("PCA_", all_genes_method_label, ".pdf"))
    ggsave(out_all_pdf, plot = p_all$plot, width = 8, height = 5)
    message("已保存全基因 PCA PDF 图: ", out_all_pdf)

    if (!is.null(p_all$plot_batch)) {
      out_all_batch_png <- file.path(output_dir, paste0("PCA_", all_genes_method_label, "_by_batch.png"))
      ggsave(out_all_batch_png, plot = p_all$plot_batch, width = 8, height = 5, dpi = 150, bg = "white")
      message("已保存全基因 batch 着色 PCA 图: ", out_all_batch_png)

      out_all_batch_pdf <- file.path(output_dir, paste0("PCA_", all_genes_method_label, "_by_batch.pdf"))
      ggsave(out_all_batch_pdf, plot = p_all$plot_batch, width = 8, height = 5)
      message("已保存全基因 batch 着色 PCA PDF 图: ", out_all_batch_pdf)
    }
  }
}

if (enable_random_100_genes_pca) {
  all_available_genes <- colnames(t_data)
  if (length(all_available_genes) < random_100_genes_n) {
    warning(
      "可用基因数 (", length(all_available_genes), ") 少于 ",
      random_100_genes_n, "，跳过随机基因 PCA。"
    )
  } else {
    random_gene_sets <- list()
    random_metrics_list <- list()
    random_plot_list <- list()
    random_plot_batch_list <- list()

    for (rep_i in seq_len(random_100_genes_replicates)) {
      method_label <- paste0(random_100_genes_method_prefix, "_rep", rep_i)
      set.seed(random_100_genes_seed + rep_i)
      sampled_genes <- sample(all_available_genes, random_100_genes_n, replace = FALSE)
      random_gene_sets[[method_label]] <- sampled_genes

      message(
        "开始随机 ", random_100_genes_n, " 基因 PCA（第 ",
        rep_i, "/", random_100_genes_replicates, " 次, seed=",
        random_100_genes_seed + rep_i, "）..."
      )
      p_rand <- run_pca_for_gene_vector(sampled_genes, method_label, info)
      if (!is.null(p_rand)) {
        metrics_list[[method_label]] <- p_rand$metrics
        random_metrics_list[[method_label]] <- p_rand$metrics
        random_plot_list[[method_label]] <- p_rand$plot

        out_rand_png <- file.path(output_dir, paste0("PCA_", method_label, ".png"))
        ggsave(out_rand_png, plot = p_rand$plot, width = 8, height = 5, dpi = 150, bg = "white")
        message("已保存随机基因 PCA 图: ", out_rand_png)

        out_rand_pdf <- file.path(output_dir, paste0("PCA_", method_label, ".pdf"))
        ggsave(out_rand_pdf, plot = p_rand$plot, width = 8, height = 5)
        message("已保存随机基因 PCA PDF 图: ", out_rand_pdf)

        if (!is.null(p_rand$plot_batch)) {
          random_plot_batch_list[[method_label]] <- p_rand$plot_batch

          out_rand_batch_png <- file.path(
            output_dir, paste0("PCA_", method_label, "_by_batch.png")
          )
          ggsave(
            out_rand_batch_png, plot = p_rand$plot_batch,
            width = 8, height = 5, dpi = 150, bg = "white"
          )
          message("已保存随机基因 batch 着色 PCA 图: ", out_rand_batch_png)

          out_rand_batch_pdf <- file.path(
            output_dir, paste0("PCA_", method_label, "_by_batch.pdf")
          )
          ggsave(out_rand_batch_pdf, plot = p_rand$plot_batch, width = 8, height = 5)
          message("已保存随机基因 batch 着色 PCA PDF 图: ", out_rand_batch_pdf)
        }
      }
    }

    if (length(random_gene_sets) > 0) {
      random_genes_long <- do.call(
        rbind,
        lapply(names(random_gene_sets), function(label) {
          data.frame(
            Method = label,
            Gene = random_gene_sets[[label]],
            stringsAsFactors = FALSE
          )
        })
      )
      random_genes_path <- file.path(output_dir, "PCA_random100_gene_sets.tsv")
      write.table(
        random_genes_long,
        file = random_genes_path,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
      )
      message("已保存随机基因列表: ", random_genes_path)
    }

    if (length(random_metrics_list) > 0) {
      random_metrics_df <- do.call(rbind, random_metrics_list)
      random_metrics_path <- file.path(output_dir, "PCA_random100_metrics_by_replicate.tsv")
      write.table(
        random_metrics_df,
        file = random_metrics_path,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
      )
      message("已保存随机基因各重复指标: ", random_metrics_path)

      numeric_cols <- c(
        "N_genes", "N_samples", "PC1_var_pct", "PC2_var_pct",
        "R2_Age_PC1_PC2", "R2_Age_PC1_PC2_adj_batch",
        "R2_Age_batch_only", "R2_PC_incremental_after_batch"
      )
      numeric_cols <- intersect(numeric_cols, colnames(random_metrics_df))
      random_summary <- data.frame(
        Metric = numeric_cols,
        Mean = sapply(numeric_cols, function(col) mean(random_metrics_df[[col]], na.rm = TRUE)),
        SD = sapply(numeric_cols, function(col) sd(random_metrics_df[[col]], na.rm = TRUE)),
        Min = sapply(numeric_cols, function(col) min(random_metrics_df[[col]], na.rm = TRUE)),
        Max = sapply(numeric_cols, function(col) max(random_metrics_df[[col]], na.rm = TRUE)),
        stringsAsFactors = FALSE
      )
      random_summary_path <- file.path(output_dir, "PCA_random100_metrics_summary.tsv")
      write.table(
        random_summary,
        file = random_summary_path,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
      )
      message("已保存随机基因指标汇总（5 次均值/标准差）: ", random_summary_path)
    }

    if (length(random_plot_list) > 0) {
      plts_rand <- random_plot_list[order(names(random_plot_list))]
      if (!requireNamespace("patchwork", quietly = TRUE)) {
        message("未检测到 patchwork，随机基因 3x2 合并图将使用 gridExtra。")
        if (!requireNamespace("gridExtra", quietly = TRUE)) {
          warning("无法拼接随机基因 3x2 图：需要 patchwork 或 gridExtra。")
        } else {
          grobs_rand <- c(plts_rand, list(grid::nullGrob()))
          out_rand_big_png <- file.path(output_dir, "PCA_random100_3x2.png")
          png(out_rand_big_png, width = 17, height = 10, units = "in", res = 150, bg = "white")
          gridExtra::grid.arrange(grobs = grobs_rand, ncol = 3, nrow = 2)
          dev.off()
          message("已保存随机基因 3x2 合并 PCA 图: ", out_rand_big_png)

          out_rand_big_pdf <- file.path(output_dir, "PCA_random100_3x2.pdf")
          pdf(out_rand_big_pdf, width = 17, height = 10, bg = "white")
          gridExtra::grid.arrange(grobs = grobs_rand, ncol = 3, nrow = 2)
          dev.off()
          message("已保存随机基因 3x2 合并 PCA PDF 图: ", out_rand_big_pdf)
        }
      } else {
        library(patchwork)
        if (length(plts_rand) >= 5) {
          combined_rand <- (plts_rand[[1]] | plts_rand[[2]] | plts_rand[[3]]) /
            (plts_rand[[4]] | plts_rand[[5]] | plot_spacer())
        } else {
          combined_rand <- Reduce(`|`, plts_rand)
        }

        out_rand_big_png <- file.path(output_dir, "PCA_random100_3x2.png")
        ggsave(out_rand_big_png, plot = combined_rand, width = 17, height = 10, dpi = 150, bg = "white")
        message("已保存随机基因 3x2 合并 PCA 图: ", out_rand_big_png)

        out_rand_big_pdf <- file.path(output_dir, "PCA_random100_3x2.pdf")
        ggsave(out_rand_big_pdf, plot = combined_rand, width = 17, height = 10)
        message("已保存随机基因 3x2 合并 PCA PDF 图: ", out_rand_big_pdf)
      }
    }

    if (length(random_plot_batch_list) > 0) {
      plts_rand_batch <- random_plot_batch_list[order(names(random_plot_batch_list))]
      if (!requireNamespace("patchwork", quietly = TRUE)) {
        if (requireNamespace("gridExtra", quietly = TRUE)) {
          grobs_rand_batch <- c(plts_rand_batch, list(grid::nullGrob()))
          out_rand_batch_big_png <- file.path(output_dir, "PCA_random100_3x2_by_batch.png")
          png(out_rand_batch_big_png, width = 17, height = 10, units = "in", res = 150, bg = "white")
          gridExtra::grid.arrange(grobs = grobs_rand_batch, ncol = 3, nrow = 2)
          dev.off()
          message("已保存随机基因 3x2 batch 合并 PCA 图: ", out_rand_batch_big_png)

          out_rand_batch_big_pdf <- file.path(output_dir, "PCA_random100_3x2_by_batch.pdf")
          pdf(out_rand_batch_big_pdf, width = 17, height = 10, bg = "white")
          gridExtra::grid.arrange(grobs = grobs_rand_batch, ncol = 3, nrow = 2)
          dev.off()
          message("已保存随机基因 3x2 batch 合并 PCA PDF 图: ", out_rand_batch_big_pdf)
        }
      } else {
        library(patchwork)
        if (length(plts_rand_batch) >= 5) {
          combined_rand_batch <- (plts_rand_batch[[1]] | plts_rand_batch[[2]] | plts_rand_batch[[3]]) /
            (plts_rand_batch[[4]] | plts_rand_batch[[5]] | plot_spacer())
        } else {
          combined_rand_batch <- Reduce(`|`, plts_rand_batch)
        }

        out_rand_batch_big_png <- file.path(output_dir, "PCA_random100_3x2_by_batch.png")
        ggsave(out_rand_batch_big_png, plot = combined_rand_batch, width = 17, height = 10, dpi = 150, bg = "white")
        message("已保存随机基因 3x2 batch 合并 PCA 图: ", out_rand_batch_big_png)

        out_rand_batch_big_pdf <- file.path(output_dir, "PCA_random100_3x2_by_batch.pdf")
        ggsave(out_rand_batch_big_pdf, plot = combined_rand_batch, width = 17, height = 10)
        message("已保存随机基因 3x2 batch 合并 PCA PDF 图: ", out_rand_batch_big_pdf)
      }
    }
  }
}

if (length(metrics_list) > 0) {
  metrics_df <- do.call(rbind, metrics_list)
  metrics_path <- file.path(output_dir, metrics_output_file)
  write.table(metrics_df, file = metrics_path, sep = "\t", quote = FALSE, row.names = FALSE)
  message("已保存年龄梯度定量评估表: ", metrics_path)
}

if (length(plot_list) == 0 && !enable_all_genes_pca) {
  stop("所有方法的基因数都不足以进行 PCA（<2 个基因），未生成任何图形。")
}
if (length(plot_list) == 0 && enable_all_genes_pca) {
  message("各模型基因集 PCA 未生成图形，但已尝试全基因 PCA。")
}

if (length(plot_list) > 0) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("未检测到 patchwork 包，将使用 gridExtra 进行拼图。")
    if (!requireNamespace("gridExtra", quietly = TRUE)) {
      stop("需要安装 'patchwork' 或 'gridExtra' 包以拼接 3x2 大图。")
    } else {
      gridExtra::grid.arrange(grobs = plot_list, ncol = 3, nrow = 2)
      out_file_big_png <- file.path(output_dir, "PCA_all_methods_3x2.png")
      ggsave(out_file_big_png, width = 17, height = 10, dpi = 150, bg = "white")

      out_file_big_pdf <- file.path(output_dir, "PCA_all_methods_3x2.pdf")
      ggsave(out_file_big_pdf, width = 17, height = 10)
    }
  } else {
    library(patchwork)
    plts <- plot_list
    n_plots <- length(plts)
    if (n_plots >= 6) {
      combined <- (plts[[1]] | plts[[2]] | plts[[3]]) /
        (plts[[4]] | plts[[5]] | plts[[6]])
    } else if (n_plots > 1) {
      combined <- Reduce(`|`, plts)
    } else {
      combined <- plts[[1]]
    }

    out_file_big_png <- file.path(output_dir, "PCA_all_methods_3x2.png")
    ggsave(out_file_big_png, plot = combined, width = 17, height = 10, dpi = 150, bg = "white")
    message("已保存 3x2 合并 PCA 图: ", out_file_big_png)

    out_file_big_pdf <- file.path(output_dir, "PCA_all_methods_3x2.pdf")
    ggsave(out_file_big_pdf, plot = combined, width = 17, height = 10)
    message("已保存 3x2 合并 PCA PDF 图: ", out_file_big_pdf)
  }
}

if (length(plot_batch_list) > 0) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    message("未检测到 patchwork，batch 合并图将使用 gridExtra。")
    if (!requireNamespace("gridExtra", quietly = TRUE)) {
      warning("无法拼接 batch 合并图：需要 patchwork 或 gridExtra。")
    } else {
      gridExtra::grid.arrange(grobs = plot_batch_list, ncol = 3, nrow = 2)
      out_batch_big_png <- file.path(output_dir, "PCA_all_methods_3x2_by_batch.png")
      ggsave(out_batch_big_png, width = 17, height = 10, dpi = 150, bg = "white")
      out_batch_big_pdf <- file.path(output_dir, "PCA_all_methods_3x2_by_batch.pdf")
      ggsave(out_batch_big_pdf, width = 17, height = 10)
    }
  } else {
    plts_batch <- plot_batch_list
    n_batch_plots <- length(plts_batch)
    if (n_batch_plots >= 6) {
      combined_batch <- (plts_batch[[1]] | plts_batch[[2]] | plts_batch[[3]]) /
        (plts_batch[[4]] | plts_batch[[5]] | plts_batch[[6]])
    } else if (n_batch_plots > 1) {
      combined_batch <- Reduce(`|`, plts_batch)
    } else {
      combined_batch <- plts_batch[[1]]
    }

    out_batch_big_png <- file.path(output_dir, "PCA_all_methods_3x2_by_batch.png")
    ggsave(out_batch_big_png, plot = combined_batch, width = 17, height = 10, dpi = 150, bg = "white")
    message("已保存 3x2 batch 合并 PCA 图: ", out_batch_big_png)

    out_batch_big_pdf <- file.path(output_dir, "PCA_all_methods_3x2_by_batch.pdf")
    ggsave(out_batch_big_pdf, plot = combined_batch, width = 17, height = 10)
    message("已保存 3x2 batch 合并 PCA PDF 图: ", out_batch_big_pdf)
  }
}
