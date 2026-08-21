#!/bin/bash

#SBATCH --job-name=featurecount
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/mnt3/wanggd_group/zt/transcriptome/featurecounts/featurecounts.out
#SBATCH --error=/mnt3/wanggd_group/zt/transcriptome/featurecounts/featurecounts.err

sam_list=`find /mnt3/wanggd_group/zt/transcriptome/hisat2/ -name "*.sam"`

featureCounts -p --countReadPairs -T 12 -t exon -g gene_id -a /mnt/wanggd_group/zt/transcriptome/dog/reference/Canis_lupus_familiaris.CanFam3.1.104.gtf -o /mnt3/wanggd_group/zt/transcriptome/featurecounts/count.txt $sam_list 

