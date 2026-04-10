library(ggvenn)

setwd("F:/scientific_research/aging_project/all_count/20260129/aging_clock/intergration5/all_coefficients")

data <- read.table("ortholog_maSigPro_61个基因.txt",head = T,sep = "\t")

set1 <- data$all_intersection
set2 <- data$maSigPro
set3 <- data$Border_collies10
set4 <- data$Beagle40

# 去掉空字符串和 NA
set1 <- set1[!(is.na(set1) | set1 == "")]
set2 <- set2[!(is.na(set2) | set2 == "")]
set3 <- set3[!(is.na(set3) | set3 == "")]
set4 <- set4[!(is.na(set4) | set4 == "")]

# a <- list(
#   'all_intersection' = set1,
#   'Zeng et al.' = set2,
#   'Jónás et al.' = set3,
#   'Koch et al.' = set4
# )

a <- list(
  'all_intersection' = set1,
  'Zeng et al.' = set2,
  'Jónás et al.' = set3
)

ggvenn(a, show_elements = FALSE, show_percentage = FALSE, digits = 1,
       fill_color = c("#ffb2b2", "#F0E442", "#b2e7cb", "#b2b2ff"), fill_alpha = 0.5,
       stroke_color = "white", stroke_alpha = 1, stroke_size = 2, stroke_linetype = "solid",
       set_name_color = c("#ffb2b2", "#F0E442", "#b2e7cb", "#b2b2ff"),
       set_name_size = 7,
       text_color = "black",
       text_size = 7)
       
# 取交集
intersection_1_2 <- intersect(set1,set2)
intersection_1_2_3 <- intersect(intersection_1_2,set3)
write.table(intersection_1_2_3, "intersection_3studiess_gene.txt",sep = "\t")
