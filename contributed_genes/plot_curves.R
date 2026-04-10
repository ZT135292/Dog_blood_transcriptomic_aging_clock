library(ggplot2)
## 如要让每个小图都显示各自的 x 轴，需要用到 ggh4x 的 facet_wrap2
## 若尚未安装，请先在 R 中运行：install.packages("ggh4x")
library(ggh4x)

setwd("F:/scientific_research/aging_project/all_count/20260129/aging_clock/intergration5/all_coefficients/")

# ========== 可选设置 ==========
use_log_expression <- FALSE   # TRUE = 表达量取 log2 后绘图；FALSE = 原始表达量
remove_outliers <- TRUE    # TRUE = 按 IQR 规则过滤每个基因的异常值；FALSE = 不过滤
output_dir <- "expression_curves_R"          # 单基因图片保存文件夹
combined_dir <- "expression_curves_R_all"      # 合并大图保存文件夹
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
dir.create(combined_dir, showWarnings = FALSE, recursive = TRUE)
message("单基因图片将保存到文件夹: ", output_dir)
message("合并大图将保存到文件夹: ", combined_dir)

# =========================
# 1) 单基因散点图 + 平滑曲线（PNG + PDF），并按基因去除异常值（与 plot_dog.R 一致）
# =========================

# 用于保存合并绘图的数据（已按基因过滤异常值）
combined_df <- data.frame(
  Age = numeric(0),
  Expression = numeric(0),
  Gene = character(0),
  stringsAsFactors = FALSE
)

for (i in seq_along(gene_col_map)) {
  gene_label <- names(gene_col_map)[i]
  col_name <- gene_col_map[i]

  # 构建单基因数据框（与 plot_dog.R 一致：Age + Expression）
  gene_df <- data.frame(
    Gene = gene_label,
    Age = info$Chronological_Age,
    Expression = t_data[[col_name]],
    stringsAsFactors = FALSE
  )

  # 可选：按 IQR 去除该基因的表达异常值（与 plot_dog.R 一致）
  if (remove_outliers) {
    qs <- quantile(gene_df$Expression, probs = c(0.25, 0.75), na.rm = TRUE)
    IQR_val <- qs[2] - qs[1]
    lower <- qs[1] - 1.5 * IQR_val
    upper <- qs[2] + 1.5 * IQR_val
    before_n <- nrow(gene_df)
    gene_df <- subset(gene_df, Expression >= lower & Expression <= upper)
    after_n <- nrow(gene_df)
    message("基因 ", gene_label, " 去掉异常值: ", before_n - after_n, " 个，保留 ", after_n, " 个样本。")
  } else {
    message("基因 ", gene_label, " 未进行异常值过滤，使用全部 ", nrow(gene_df), " 个样本。")
  }

  # 若该基因所有点都被过滤，则跳过
  if (nrow(gene_df) == 0) {
    message("基因 ", gene_label, " 在去除异常值后无数据，跳过绘图。")
    next
  }

  # 追加到合并数据中（用于后面 facet 大图）
  combined_df <- rbind(combined_df, gene_df[, c("Age", "Expression", "Gene")])

  y_lab <- if (use_log_expression) "Expression (log2)" else "TMM + CPM"
  p <- ggplot(gene_df, aes(x = Age, y = Expression)) +
    geom_smooth(color = "red", se = TRUE, fill = "steelblue", alpha = 0.3) +
    labs(title = paste0(gene_label, " Expression"),
         x = "Age of Dogs",
         y = y_lab)

  p <- p + theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_continuous(breaks = c(1, 3, 5, 7, 9, 11, 13, 15),
                       labels = c("1", "3", "5", "7", "9", "11", "13", "15")) +
    theme(plot.title = element_text(size = 18, face = "bold"),
          axis.title = element_text(size = 16, face = "bold"),
          axis.text = element_text(size = 14),
          axis.line = element_line(color = "black"),      # 显示坐标轴线
          axis.ticks = element_line(color = "black"),     # 显示刻度线
          plot.background = element_rect(fill = "transparent", color = NA),
          panel.background = element_rect(fill = "transparent", color = NA))

  # 保存 PNG
  out_png <- file.path(output_dir, paste0("expression_curve_", gene_label, ".png"))
  ggsave(out_png, plot = p, width = 7, height = 4, dpi = 150, bg = "transparent")

  # 保存 PDF
  out_pdf <- file.path(output_dir, paste0("expression_curve_", gene_label, ".pdf"))
  ggsave(out_pdf, plot = p, width = 7, height = 4, device = "pdf", bg = "transparent")

  message("已保存单基因 PNG: ", out_png)
  message("已保存单基因 PDF: ", out_pdf)
}

# =========================
# 2) 所有基因合并到一张大图（facet），并保存为 PDF/PNG
#    使用同样“去除异常值后”的 combined_df 数据（与 plot_dog.R 逻辑一致）
# =========================

if (nrow(combined_df) > 0) {
  y_lab <- if (use_log_expression) "Expression (log2)" else "TMM + CPM"

  p_all <- ggplot(combined_df, aes(x = Age, y = Expression)) +
    geom_smooth(color = "red", se = TRUE, fill = "steelblue", alpha = 0.3) +
    labs(title = "Expression Curves of All Genes",
         x = "Age of Dogs",
         y = y_lab) +
    ## 使用 ggh4x::facet_wrap2，并设置 axes = "x"，让每个 panel 都绘制自己的 x 轴
    facet_wrap2(~ Gene, scales = "free_y", ncol = 3, nrow = 4, axes = "x") +
    theme(plot.title = element_text(hjust = 0.5)) +
    scale_x_continuous(breaks = c(1, 3, 5, 7, 9, 11, 13, 15),
                       labels = c("1", "3", "5", "7", "9", "11", "13", "15")) +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.title = element_text(size = 18, face = "bold"),
      axis.text = element_text(size = 16, face = "bold"),
      axis.line = element_line(color = "black"),        # 显示坐标轴线
      axis.ticks = element_line(color = "black"),       # 显示刻度线
      strip.text = element_text(size = 16, face = "bold"),
      strip.background = element_blank(),      # 去掉基因名背景灰条
      plot.background = element_rect(fill = "transparent", color = NA),
      panel.background = element_rect(fill = "white", color = NA),   # 面板背景设为白色
      panel.spacing.x = grid::unit(0.1, "lines"),   # 左右间距保持很小
      panel.spacing.y = grid::unit(0.9, "lines")    # 增加上下间距，避免下图遮挡上图 x 轴
    )

  # 保存合并大图 PNG
  combined_png <- file.path(combined_dir, "expression_curves_all_genes.png")
  ggsave(combined_png, plot = p_all, width = 14, height = 8, dpi = 150, bg = "white")  # PNG 背景设为白色

  # 保存合并大图 PDF
  combined_pdf <- file.path(combined_dir, "expression_curves_all_genes.pdf")
  ggsave(combined_pdf, plot = p_all, width = 14, height = 8, device = "pdf", bg = "transparent")

  message("已保存合并 PNG: ", combined_png)
  message("已保存合并 PDF: ", combined_pdf)
} else {
  message("所有基因在去除异常值后均无数据，未生成合并大图。")
}