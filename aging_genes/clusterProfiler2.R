library(clusterProfiler)
library(org.Cf.eg.db)
library(DOSE)
library(ggplot2)
library(ggrepel) 
library(stringr) 
library(dplyr)

R.utils::setOption("clusterProfiler.download.method","auto")

setwd("F:/scientific_research/aging_project/all_count/20260723/no_group/all_coefficients/")

truncate_description <- function(x, n_words = 7) {
  sapply(strsplit(x, "\\s+"), function(words) {
    words <- words[words != ""]
    if (length(words) <= n_words) {
      paste(words, collapse = " ")
    } else {
      paste(c(words[1:n_words], "..."), collapse = " ")
    }
  })
}

gene_list_df <- read.table("coefficients_HPO_all_top100.csv", header = TRUE, sep = ",", check.names = FALSE)
bg_genes <- data.frame(
  gene_name = rownames(
    read.table("./total_count_sort_ortholog_TMM_logcpm_normal_ref.txt",
               header = TRUE, row.names = 1, sep = "\t", check.names = FALSE)
  )
)

output_dir <- "enrichment_results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

bg_genes_list <- bitr(
  bg_genes$gene_name,
  fromType = "SYMBOL",
  toType   = c("ENSEMBL","ENTREZID","SYMBOL"),
  OrgDb    = "org.Cf.eg.db"
)

run_enrichment_for_gene_set <- function(gene_symbols, set_name, bg_genes_list) {
  gene_symbols <- unique(na.omit(trimws(gene_symbols)))
  gene_symbols <- gene_symbols[gene_symbols != ""]
  if (length(gene_symbols) == 0) {
    message("基因集 ", set_name, " 为空，跳过。")
    return(NULL)
  }
  
  message("=== 开始基因集富集分析: ", set_name, " (", length(gene_symbols), " genes) ===")
  
  conv <- bitr(
    gene_symbols,
    fromType = "SYMBOL",
    toType   = c("ENSEMBL","ENTREZID","SYMBOL"),
    OrgDb    = "org.Cf.eg.db"
  )
  if (is.null(conv) || nrow(conv) == 0) {
    message("  - 基因集 ", set_name, " 在注释库中未找到有效基因，跳过。")
    return(NULL)
  }
  
  go_res <- enrichGO(
    gene          = conv$SYMBOL,
    OrgDb         = "org.Cf.eg.db",
    keyType       = "SYMBOL",
    universe      = bg_genes_list$SYMBOL,
    ont           = "ALL",
    pAdjustMethod = "BH",
    pvalueCutoff  = 1,
    qvalueCutoff  = 1,
    readable      = TRUE
  )
  
  if (is.null(go_res) || nrow(as.data.frame(go_res)) == 0) {
    message("  - GO 结果为空。")
  } else {
    go_df <- as.data.frame(go_res)
    out_go_txt <- file.path(output_dir, paste0("GO_", set_name, "_results.txt"))
    write.table(go_df, out_go_txt, sep = "\t", row.names = FALSE, quote = FALSE)
    
    go_plot_df <- go_df %>%
      filter(ONTOLOGY == "BP") %>%
      mutate(p.adjust = as.numeric(p.adjust)) %>%
      arrange(p.adjust) %>%
      head(20) %>%
      mutate(
        Description = truncate_description(Description, n_words = 7),
        Description = str_wrap(Description, width = 50),
        Description = factor(Description, levels = unique(rev(Description)))
      )
    
    if (nrow(go_plot_df) > 0) {
      p_go <- ggplot(go_plot_df, aes(x = Description, y = Count, fill = p.adjust)) +
        geom_col() +
        coord_flip() +
        scale_fill_gradient(low = "#E5968E", high = "#3B77A8") +
        labs(
          x = "GO Biological Process",
          y = "Gene Count",
          title = paste0("Top 20 BP GO Terms (", set_name, ")"),
          fill = "p.adjust"
        ) +
        theme_bw() +
        theme(axis.text = element_text(color = "black"))
      
      ggsave(file.path(output_dir, paste0("GO_barplot_", set_name, ".pdf")),
             plot = p_go, width = 10, height = 8)
    }
  }
  
  if (!"ENTREZID" %in% colnames(conv)) {
    message("  - 无 ENTREZID 信息，跳过 KEGG。")
  } else {
    kegg_res <- enrichKEGG(
      gene     = conv$ENTREZID,
      organism = "cfa",
      universe = bg_genes_list$ENTREZID,
      pvalueCutoff = 1
    )
    
    if (!is.null(kegg_res) && nrow(as.data.frame(kegg_res)) > 0) {
      kegg_df <- as.data.frame(kegg_res)
      out_kegg_txt <- file.path(output_dir, paste0("KEGG_", set_name, "_results.txt"))
      write.table(kegg_df, out_kegg_txt, sep = "\t", row.names = FALSE, quote = FALSE)
      
      kegg_plot_df <- kegg_df %>%
        mutate(pvalue = as.numeric(pvalue)) %>%
        arrange(pvalue) %>%
        head(20) %>%
        mutate(
          Description   = truncate_description(Description, n_words = 7),
          Description   = str_wrap(Description, width = 50),
          GeneRatio_num = as.numeric(sapply(strsplit(GeneRatio, "/"), function(x) as.numeric(x[1]) / as.numeric(x[2]))),
          Description   = factor(Description, levels = unique(rev(Description)))
        )
      
      if (nrow(kegg_plot_df) > 0) {
        p_kegg <- ggplot(kegg_plot_df, aes(x = GeneRatio_num, y = Description)) +
          geom_point(aes(size = Count, color = p.adjust)) +
          scale_color_gradient(low = "red", high = "blue") +
          labs(
            x = "GeneRatio",
            y = "KEGG Pathway",
            title = paste0("Top 20 KEGG Enrichment (", set_name, ")"),
            size = "Gene Count"
          ) +
          theme_bw() +
          theme(axis.text = element_text(color = "black"))
        
        ggsave(file.path(output_dir, paste0("KEGG_bubbleplot_", set_name, ".pdf")),
               plot = p_kegg, width = 10, height = 8)
      }
    } else {
      message("  - KEGG 结果为空。")
    }
  }
  
  message("=== 基因集 ", set_name, " 富集分析完成 ===\n")
}

method_names <- colnames(gene_list_df)
method_names <- method_names[!is.na(method_names) & nchar(trimws(method_names)) > 0]

for (m in method_names) {
  genes_m <- gene_list_df[[m]]
  run_enrichment_for_gene_set(genes_m, m, bg_genes_list)
}

message("全部列的单独 GO / KEGG 富集分析完成！")


library(readr)

message("开始汇总各模型 GO/KEGG 结果并绘制对比图...")

go_files <- list.files(output_dir, pattern = "^GO_.*_results\\.txt$", full.names = TRUE)

all_go_list <- list()
for (f in go_files) {
  fname <- basename(f)
  method <- sub("^GO_(.*)_results\\.txt$", "\\1", fname)
  df <- read.table(f, header = TRUE, sep = "\t", check.names = FALSE, quote = "", comment.char = "", stringsAsFactors = FALSE)
  df$method <- method
  all_go_list[[length(all_go_list) + 1]] <- df
}

if (length(all_go_list) > 0) {
  all_go <- do.call(rbind, all_go_list)
  
  go_top <- all_go %>%
    filter(ONTOLOGY == "BP") %>%
    mutate(p.adjust = as.numeric(p.adjust)) %>%
    group_by(method) %>%
    arrange(p.adjust) %>%
    slice_head(n = 5) %>%
    ungroup() %>%
    mutate(
      Description = truncate_description(Description, n_words = 7),
      Description = str_wrap(Description, width = 60)
    )
  
  go_top <- go_top %>%
    mutate(method = factor(method, levels = method_names)) %>%
    group_by(method) %>%
    arrange(p.adjust, Description, .by_group = TRUE) %>%
    mutate(plot_order = row_number()) %>%
    ungroup() %>%
    arrange(method, plot_order)
  
  out_go_top_txt <- file.path(output_dir, "GO_BP_top5_each_model.txt")
  write.table(go_top, out_go_top_txt, sep = "\t", row.names = FALSE, quote = FALSE)
  
  if (nrow(go_top) > 0) {
    # Keep table row 1 (most significant) at the top of the y-axis.
    go_top <- go_top %>%
      mutate(Description = factor(Description, levels = rev(unique(Description))))
    
    p_go_all <- ggplot(go_top, aes(x = method, y = Description)) +
      geom_point(aes(size = Count, color = -log10(p.adjust))) +
      scale_color_gradient(low = "blue", high = "red") +
      labs(
        x = "Model",
        y = "GO Biological Process",
        title = "Top GO BP Terms Across Models",
        size = "Gene Count",
        color = "-log10(p.adjust)"
      ) +
      theme_bw(base_size = 16) +
      theme(
        axis.text.x  = element_text(angle = 45, hjust = 1, size = 14),
        axis.text.y  = element_text(size = 14),
        axis.title   = element_text(size = 16),
        plot.title   = element_text(size = 18, face = "bold"),
        legend.title = element_text(size = 14),
        legend.text  = element_text(size = 13)
      )
    
    ggsave(file.path(output_dir, "GO_BP_dotplot_all_models.pdf"),
           plot = p_go_all, width = 10, height = 8)
  } else {
    message("GO 汇总后没有可用于绘图的 BP 条目。")
  }
} else {
  message("未找到任何 GO_*_results.txt 文件，跳过 GO 汇总绘图。")
}


kegg_files <- list.files(output_dir, pattern = "^KEGG_.*_results\\.txt$", full.names = TRUE)

all_kegg_list <- list()
for (f in kegg_files) {
  fname <- basename(f)
  method <- sub("^KEGG_(.*)_results\\.txt$", "\\1", fname)
  df <- read.table(f, header = TRUE, sep = "\t", check.names = FALSE, quote = "", comment.char = "", stringsAsFactors = FALSE)
  df$method <- method
  all_kegg_list[[length(all_kegg_list) + 1]] <- df
}

if (length(all_kegg_list) > 0) {
  all_kegg <- do.call(rbind, all_kegg_list)
  
  kegg_top <- all_kegg %>%
    mutate(pvalue = as.numeric(pvalue)) %>%
    group_by(method) %>%
    arrange(pvalue) %>%
    slice_head(n = 5) %>%
    ungroup() %>%
    mutate(
      Description   = truncate_description(Description, n_words = 7),
      Description   = str_wrap(Description, width = 60),
      GeneRatio_num = as.numeric(sapply(strsplit(GeneRatio, "/"), function(x) as.numeric(x[1]) / as.numeric(x[2])))
    )
  
  kegg_top <- kegg_top %>%
    mutate(method = factor(method, levels = method_names)) %>%
    group_by(method) %>%
    arrange(pvalue, Description, .by_group = TRUE) %>%
    mutate(plot_order = row_number()) %>%
    ungroup() %>%
    arrange(method, plot_order)
  
  out_kegg_top_txt <- file.path(output_dir, "KEGG_top5_each_model.txt")
  write.table(kegg_top, out_kegg_top_txt, sep = "\t", row.names = FALSE, quote = FALSE)
  
  if (nrow(kegg_top) > 0) {
    kegg_top <- kegg_top %>%
      mutate(Description = factor(Description, levels = rev(unique(Description))))
    
    p_kegg_all <- ggplot(kegg_top, aes(x = method, y = Description)) +
      geom_point(aes(size = GeneRatio_num, color = -log10(pvalue))) +
      scale_color_gradient(low = "blue", high = "red") +
      labs(
        x = "Model",
        y = "KEGG Pathway",
        title = "Top KEGG Pathways Across Models",
        size = "GeneRatio",
        color = "-log10(pvalue)"
      ) +
      theme_bw(base_size = 16) +
      theme(
        axis.text.x  = element_text(angle = 45, hjust = 1, size = 14),
        axis.text.y  = element_text(size = 14),
        axis.title   = element_text(size = 16),
        plot.title   = element_text(size = 18, face = "bold"),
        legend.title = element_text(size = 14),
        legend.text  = element_text(size = 13)
      )
    
    ggsave(file.path(output_dir, "KEGG_dotplot_all_models.pdf"),
           plot = p_kegg_all, width = 10, height = 8)
  } else {
    message("KEGG 汇总后没有可用于绘图的条目。")
  }
} else {
  message("未找到任何 KEGG_*_results.txt 文件，跳过 KEGG 汇总绘图。")
}

message("全部列的 GO / KEGG 富集分析与模型对比图绘制完成！")