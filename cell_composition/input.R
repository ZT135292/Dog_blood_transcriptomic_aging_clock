library(Seurat)
library(dplyr)

# Build a balanced CIBERSORTx reference from GSE225599 (7 leukocyte classes).
seu.obj.h <- readRDS(file = "./GSE225599_final_dataSet_H.rds")

# Rare/ambiguous populations are excluded; they can make the signature unstable.
major_types <- c(
  "CD4 T cell",
  "CD8/NK cell",
  "B cell",
  "Monocyte",
  "Neutrophil",
  "Granulocyte",
  "DC"
)

cell_labels <- as.character(seu.obj.h$celltype.l1)
names(cell_labels) <- colnames(seu.obj.h)

counts_before <- table(cell_labels, useNA = "ifany")
print(counts_before)

missing_types <- setdiff(major_types, names(counts_before))
if (length(missing_types) > 0) {
  stop(
    "The following expected cell types are absent from celltype.l1: ",
    paste(missing_types, collapse = ", ")
  )
}

# Split by celltype.l1 rather than the active Seurat identity used by subset(..., downsample=).
cells_by_type <- split(
  names(cell_labels)[cell_labels %in% major_types],
  cell_labels[cell_labels %in% major_types]
)
cells_by_type <- cells_by_type[major_types]

# Cap at 100 cells per type so the CIBERSORTx upload stays balanced and small.
target_n <- min(100L, min(lengths(cells_by_type)))
if (target_n < 20L) {
  stop(
    "At least one major cell type has fewer than 20 cells (minimum = ",
    target_n,
    "). Reconsider the selected cell types before building the reference."
  )
}

set.seed(12)
selected_cells <- unlist(
  lapply(cells_by_type, sample, size = target_n, replace = FALSE),
  use.names = FALSE
)
seu.obj.h <- subset(seu.obj.h, cells = selected_cells)

# Convert celltype.l1 to character so unused factor levels are dropped.
counts_after <- table(as.character(seu.obj.h$celltype.l1))
print(counts_after)
stopifnot(
  setequal(names(counts_after), major_types),
  all(as.integer(counts_after) == target_n)
)

DefaultAssay(seu.obj.h) <- "RNA"
seu.obj.h <- NormalizeData(seu.obj.h)

cntMatrix <- GetAssayData(
  object = seu.obj.h,
  assay = "RNA",
  layer = "data"
)

output_labels <- as.character(
  seu.obj.h$celltype.l1[match(colnames(cntMatrix), colnames(seu.obj.h))]
)
stopifnot(!anyNA(output_labels))
colnames(cntMatrix) <- output_labels

cntMatrix <- as.data.frame(as.matrix(cntMatrix), check.names = FALSE)
cntMatrix$gene <- rownames(cntMatrix)
cntMatrix <- cntMatrix %>% relocate(gene)

output_file <- "./cibersort_forRef_major7_balanced.txt"
write.table(
  cntMatrix,
  file = output_file,
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE,
  sep = "\t"
)

write.csv(
  data.frame(
    cell_type = major_types,
    cells_before = as.integer(counts_before[major_types]),
    cells_exported = rep(target_n, length(major_types))
  ),
  file = "./cibersort_reference_cell_counts.csv",
  row.names = FALSE
)

message("Reference written to: ", output_file)
message("Cells exported per type: ", target_n)
