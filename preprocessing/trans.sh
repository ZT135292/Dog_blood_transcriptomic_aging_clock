#!/bin/bash

#SBATCH --job-name=trans
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/mnt3/wanggd_group/zt/transcriptome/trans.out
#SBATCH --error=/mnt3/wanggd_group/zt/transcriptome/trans.err

echo "Start time: `date`"

SRA_ROOT_DIR="/mnt3/wanggd_group/zt/transcriptome/"

if ! command -v fastq-dump &> /dev/null
then
    echo "fastq-dump 命令未找到。请确保SRA Toolkit已正确安装并添加到PATH中。"
    echo "你可以通过 conda 安装：conda install -c bioconda sra-tools"
    exit 1
fi

echo "开始批量转换 .sra 文件到 .fastq.gz"
echo "源目录: $SRA_ROOT_DIR"

for dir in "$SRA_ROOT_DIR"/*/; do
    if [ -d "$dir" ]; then
        echo "进入目录: $dir"
        for sra_file in "$dir"/*.sra; do
            if [ -f "$sra_file" ]; then
                echo "发现SRA文件: $sra_file"
                filename=$(basename "$sra_file" .sra)
                output_dir="$dir"
                echo "正在转换 $sra_file 到 $output_dir/${filename}.fastq.gz"
                fastq-dump --gzip --split-files -O "$output_dir" "$sra_file"
                if [ $? -eq 0 ]; then
                    echo "成功转换 $sra_file"
                else
                    echo "转换 $sra_file 失败，请检查错误信息。"
                fi
            else
                echo "在 $dir 中未找到 .sra 文件。"
            fi
        done
    fi
done

echo "所有 .sra 文件转换完成。"
echo "End time: `date`"
