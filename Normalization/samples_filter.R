# 设置工作路径
setwd("F:/scientific_research/aging_project/all_count/20260113")

# 读取数据（假设文件是制表符分隔）
data <- read.delim("total_count.txt", row.names = 1)  # 第一列是基因名
meta_data <- read.delim("total_metadata.txt", row.names = 1)

data_filtered <- data[, row.names(meta_data)]

# 写入文件（制表符分隔）
write.table(data_filtered, "total_count_sort.txt", sep = "\t", row.names = T, quote = FALSE)

# 去除含有 NA 值的行
data_filtered_no_na <- na.omit(data_filtered)

# 写入文件（制表符分隔）
write.table(data_filtered_no_na, "total_count_sort_noNa.txt", sep = "\t", row.names = T, quote = FALSE)


# 2. 对 cpm 值进行对数转换 (log2)，通常会加一个小的伪计数（如 +1）以避免 log(0)
# 这里使用 log2(x + 1)
cpm_log2 <- log2(data_filtered_no_na + 1)

# 查看对数转换后的 cpm 结果
cat("\nLog2(cpm+1) 值的头部：\n")
head(cpm_log2)

# 写入对数转换后的 cpm 文件（制表符分隔）
write.table(cpm_log2, "total_count_sort_noNa_log2.txt", sep = "\t", row.names = T, quote = FALSE)

cat("\n所有操作完成，cpm 值已保存到 'total_cpm_ortholog_filtered_noNa_log2.txt'，Log2(cpm+1) 值已保存到 'total_cpm_ortholog_healthy_noNa_log2.txt'。\n")