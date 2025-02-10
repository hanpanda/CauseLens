#!/bin/bash

download_dir="/root/lqh/multimodal-RCA/datasets/aiops_2022"

date_list=("2022-03-19" "2022-03-20" "2022-03-21" "2022-03-22" "2022-03-23" "2022-03-24" "2022-03-25" "2022-03-26" "2022-03-27" "2022-03-28" "2022-03-29" "2022-03-30" "2022-03-31" "2022-04-01" "2022-04-02")

# 循环下载文件直到结束日期
for current_date in "${date_list[@]}"; do
    # 构建URL
    tar_gz_filename="${current_date}-cloudbed1.tar.gz"
    url="https://dataset.aiops-challenge.com/dataset/2022AIOpsChallengeDATASET/${current_date}-cloudbed1.tar.gz"
    json_filename="groundtruth-k8s-1-${current_date}.json"
    json_url="https://dataset.aiops-challenge.com/dataset/2022AIOpsChallengeDATASET/groundtruth/second_label_json/groundtruth-k8s-1-${current_date}.json"
    json_url_2="https://dataset.aiops-challenge.com/dataset/2022AIOpsChallengeDATASET/groundtruth/second_label_json/second_label_json27-44/groundtruth-k8s-1-${current_date}.json"
    
    # 使用wget下载文件
    if [ ! -f "$download_dir/$tar_gz_filename" ]; then
        wget --no-check-certificate --directory-prefix="$download_dir" "$url"
    else
        echo "File $tar_gz_filename already exists, skipping download."
    fi

    if [ ! -f "$download_dir/$json_filename" ]; then
        wget --no-check-certificate --directory-prefix="$download_dir" "$json_url"
        wget --no-check-certificate --directory-prefix="$download_dir" "$json_url_2"
    else
        echo "File $json_filename already exists, skipping download."
    fi
done

wget --no-check-certificate --directory-prefix="$download_dir" https://dataset.aiops-challenge.com/dataset/2022aiops%E6%8C%91%E6%88%98%E8%B5%9B%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE/training_data_with_faults.zip
wget --no-check-certificate --directory-prefix="$download_dir" https://dataset.aiops-challenge.com/dataset/2022aiops%E6%8C%91%E6%88%98%E8%B5%9B%E8%AE%AD%E7%BB%83%E6%95%B0%E6%8D%AE/training_data_normal.tar.gz