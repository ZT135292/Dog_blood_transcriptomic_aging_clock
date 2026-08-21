#!/bin/bash

#SBATCH --job-name=seqkit
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/mnt3/wanggd_group/zt/transcriptome/seqkit.out
#SBATCH --error=/mnt3/wanggd_group/zt/transcriptome/seqkit.err

echo "Start time: `date`"

WORK_DIR="/mnt3/wanggd_group/zt/transcriptome"
OUT_FILE="${WORK_DIR}/fastq_quality_summary.tsv"

echo "开始使用 SeqKit 统计数据质量..."
seqkit stats -a -T -j 12 ${WORK_DIR}/SRR*/SRR*_[12].fastq.gz > ${OUT_FILE}
echo "统计完成！结果已保存至: ${OUT_FILE}"
echo "End time: `date`"
