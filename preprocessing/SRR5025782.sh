#!/bin/bash

#SBATCH --job-name=SRR5025782
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/mnt3/wanggd_group/zt/transcriptome/hisat2/scripts/SRR5025782.out
#SBATCH --error=/mnt3/wanggd_group/zt/transcriptome/hisat2/scripts/SRR5025782.err

hisat2 -p 4 -x /mnt/wanggd_group/zt/transcriptome/dog/reference/CanFam3_1/CanFam3_1 -1 /mnt3/wanggd_group/zt/transcriptome/SRR5025782/SRR5025782_1.fastq.gz -2 /mnt3/wanggd_group/zt/transcriptome/SRR5025782/SRR5025782_2.fastq.gz -S /mnt3/wanggd_group/zt/transcriptome/hisat2/SRR5025782.sam