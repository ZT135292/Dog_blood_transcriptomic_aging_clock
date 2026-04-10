library("edgeR")

# 设置工作路径
setwd("F:/scientific_research/aging_project/all_count/20260113")

data <- read.table("total_count_sort_ortholog_batched.txt", header = T)
col_data <- read.table("total_metadata.txt",header = T)

x <- data
group <- factor(col_data$age)
y <- DGEList(counts=x,group=group)

# keep <- filterByExpr(y, min.count = 0, min.total.count = 10)
# y <- y[keep, , keep.lib.sizes=FALSE]

y <- calcNormFactors(y, method = "TMM")
plotMDS(y)  #根据样本间的leading biological coefficient of variation (BCV)画的图 

normal_data <- cpm(y, normalized.lib.sizes = T, log = F)
  
write.table(normal_data,"total_count_sort_ortholog_batched_TMM.txt",sep = "\t",row.names = T)

