library(sva)

# 设置工作路径
setwd("F:/scientific_research/aging_project/all_count/20260113")

my_data <- read.delim("total_count_sort_ortholog_TMM.txt", row.names = 1, check.names = FALSE)
meta_data <- read.delim("total_metadata.txt", row.names = 1, check.names = FALSE)

my_data <- as.matrix(my_data)

# 您可以再次运行检查，这次结果应该是 TRUE
all(colnames(my_data) == rownames(meta_data))
# --------------------

# # 删除0元素大于1/3个的行
# n0 <- apply(my_data == 0, 1, sum) # 统计矩阵中每行为0的元素个数
# i0 <- which(n0 > ncol(my_data)*2/3) # 统计0元素大于1/3个的行号
# my_data_filter <- my_data[-i0, ] # 删除0元素大于1/3个的行

batch <- meta_data$batch
mod <- model.matrix(~ Chronological_Age, data = meta_data)

adjusted <- ComBat(my_data, batch=batch , mod = mod)

write.table(adjusted,"total_count_sort_ortholog_TMM_batched.txt", sep = "\t",  quote = FALSE, 
            row.names = TRUE, col.names = NA)