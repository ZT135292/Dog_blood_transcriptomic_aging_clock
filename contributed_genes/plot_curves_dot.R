library(ggplot2)
# 若需要让每个小图都显示 x 轴，请先安装并加载 ggh4x：
# install.packages("ggh4x")  # 只需安装一次
library(ggh4x)

setwd("F:/scientific_research/aging_project/all_count/20260129/aging_clock/intergration5/all_coefficients/")

# ========== 可选设置 ==========
use_log_expression <- FALSE   # TRUE = 表达量取 log2 后绘图；FALSE = 原始表达量
remove_outliers <- TRUE      # TRUE = 去掉异常值（按每个基因表达量 IQR 规则）
output_dir <- "expression_curves_R_dot"  # 单基因图片保存文件夹名
combine_output_prefix <- "all_genes_expression" # 合并大图文件名前缀（不含扩展名）
# ==============================

# 读取表达矩阵与样本信息
data <- read.table("total_count_sort_ortholog_TMM_batched.txt", header = TRUE, row.names = 1)
info <- read.table("total_metadata.txt", header = TRUE)

# 表达矩阵转置：行为样本，列为基因
t_data <- as.data.frame(t(data))

# 可选：表达量取对数（log2），若有 0 或负值则先平移
if (use_log_expression) {
  min_val <- min(as.matrix(t_data), na.rm = TRUE)
  offset <- if (min_val < 1) 1 - min_val else 0
  t_data <- as.data.frame(log2(as.matrix(t_data) + offset))
  message("表达量已转换为 log2(value + ", offset, ")")
}

# 读取要绘制的基因列表（列名为 intersection）
gene_list_df <- read.table("column_intersection_result.txt", header = TRUE, sep = "\t")
genes <- as.character(gene_list_df$intersection)
genes <- genes[!is.na(genes) & nchar(trimws(genes)) > 0 & tolower(trimws(genes)) != "intersection"]

# 若表达矩阵列名带前缀如 "4605:GCNT4"，用冒号后部分匹配
col_names <- colnames(t_data)
match_gene_col <- function(g) {
  g <- trimws(g)
  if (g %in% col_names) return(g)
  suffix_match <- which(sapply(strsplit(col_names, ":"), function(x) tail(x, 1)) == g)
  if (length(suffix_match) > 0) return(col_names[suffix_match[1]])
  return(NULL)
}

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
if (length(gene_col_map) == 0) stop("表达矩阵中未找到任何 intersection 中的基因，请检查基因名或列名。")
message("共匹配到 ", length(gene_col_map), " 个基因用于绘图。")

# 创建输出文件夹
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
message("图片将保存到文件夹: ", output_dir)

# 统一的 y 轴标签
y_lab <- if (use_log_expression) "Expression (log2)" else "Expression"

# 用于保存所有基因数据的长表格式数据框
all_df <- data.frame()
all_df_raw <- data.frame()  # 包含异常值，用于在合并大图中把异常点画成灰色

# 散点图 + 平滑曲线，按基因循环，单独保存每个基因的 PNG 和 PDF
for (i in seq_along(gene_col_map)) {
  gene_label <- names(gene_col_map)[i]
  col_name <- gene_col_map[i]

  # 单基因原始数据
  gene_df_raw <- data.frame(
    Gene = gene_label,
    Age = info$Chronological_Age,
    Expression = t_data[[col_name]]
  )

  # 标记异常值（IQR 规则），并根据 remove_outliers 决定是否从拟合中剔除
  gene_df <- gene_df_raw
  gene_df$is_outlier <- FALSE
  if (remove_outliers) {
    qs <- quantile(gene_df$Expression, probs = c(0.25, 0.75), na.rm = TRUE)
    IQR_val <- qs[2] - qs[1]
    lower <- qs[1] - 1.5 * IQR_val
    upper <- qs[2] + 1.5 * IQR_val
    gene_df$is_outlier <- gene_df$Expression < lower | gene_df$Expression > upper
    before_n <- nrow(gene_df)
    after_n <- sum(!gene_df$is_outlier)
    message("基因 ", gene_label, " 标记异常值: ", before_n - after_n, " 个，保留 ", after_n, " 个样本用于拟合。")
  } else {
    message("基因 ", gene_label, " 未进行异常值过滤，全部点参与拟合。")
  }

  # 若全部为异常值（极端情况），则跳过拟合与绘图
  if (all(gene_df$is_outlier)) {
    message("基因 ", gene_label, " 所有点均被视为异常值，跳过绘图。")
    next
  }

  # all_df 只保留用于拟合的点（非异常值），与原逻辑一致
  all_df <- rbind(all_df, subset(gene_df, !is_outlier)[, c("Gene", "Age", "Expression")])
  # all_df_raw 保存全部点并带有异常值标记，用于合并大图中区分颜色
  gene_df_raw$is_outlier <- gene_df$is_outlier
  all_df_raw <- rbind(all_df_raw, gene_df_raw)

  p <- ggplot(gene_df, aes(x = Age, y = Expression)) +
    # 异常值点画成灰色，正常点保持默认颜色（黑色）
    geom_point(aes(color = is_outlier)) +
    geom_smooth(data = subset(gene_df, !is_outlier),
                aes(x = Age, y = Expression),
                color = "red", se = FALSE, inherit.aes = FALSE) +
    scale_color_manual(values = c(`FALSE` = "black", `TRUE` = "grey70"),
                       name = "Outlier",
                       labels = c(`FALSE` = "Normal", `TRUE` = "Outlier")) +
    labs(title = paste0(gene_label, " Expression"),
         x = "Age of Dogs",
         y = y_lab) +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_continuous(breaks = c(1, 3, 5, 7, 9, 11, 13, 15),
                       labels = c("1", "3", "5", "7", "9", "11", "13", "15")) +
    theme(plot.title = element_text(size = 18, face = "bold"),
          axis.title = element_text(size = 16, face = "bold"),
          axis.text = element_text(size = 14))

  # 单基因 PNG
  out_png <- file.path(output_dir, paste0("expression_curve_", gene_label, ".png"))
  ggsave(out_png, plot = p, width = 7, height = 4, dpi = 150)
  # 单基因 PDF
  out_pdf <- file.path(output_dir, paste0("expression_curve_", gene_label, ".pdf"))
  ggsave(out_pdf, plot = p, width = 7, height = 4)

  message("已保存 PNG: ", out_png)
  message("已保存 PDF: ", out_pdf)
}

# 如果有至少一个基因，则绘制合并大图（3x3 小图排列）
if (nrow(all_df) > 0) {
  all_df$Gene <- factor(all_df$Gene, levels = names(gene_col_map))
  all_df_raw$Gene <- factor(all_df_raw$Gene, levels = names(gene_col_map))

  p_all <- ggplot() +
    # 合并大图中：异常值点画成灰色，正常点为黑色
    geom_point(data = all_df_raw,
               aes(x = Age, y = Expression, color = is_outlier)) +
    geom_smooth(data = all_df,
                aes(x = Age, y = Expression),
                color = "red", se = FALSE) +
    scale_color_manual(values = c(`FALSE` = "black", `TRUE` = "grey70"),
                       name = "Outlier",
                       labels = c(`FALSE` = "Normal", `TRUE` = "Outlier")) +
    labs(title = "Gene Expression vs Age",
         x = "Age of Dogs",
         y = y_lab) +
    # 使用 ggh4x::facet_wrap2，让每个小图都显示 x 轴（axes = "x"）
    facet_wrap2(~ Gene, ncol = 3, scales = "free_y", axes = "x") +
    theme(plot.title = element_text(hjust = 0.5),
          strip.text = element_text(size = 12, face = "bold")) +
    scale_x_continuous(breaks = c(1, 3, 5, 7, 9, 11, 13, 15),
                       labels = c("1",  "3",  "5",  "7",  "9",  "11",  "13",  "15")) +
    theme(axis.title = element_text(size = 12, face = "bold"),
          axis.text = element_text(size = 10))

  # 合并大图 PNG
  out_all_png <- file.path(output_dir, paste0(combine_output_prefix, ".png"))
  ggsave(out_all_png, plot = p_all, width = 16, height = 9, dpi = 150)

  # 合并大图 PDF
  out_all_pdf <- file.path(output_dir, paste0(combine_output_prefix, ".pdf"))
  ggsave(out_all_pdf, plot = p_all, width = 16, height = 9)

  message("已保存合并大图 PNG: ", out_all_png)
  message("已保存合并大图 PDF: ", out_all_pdf)
}
