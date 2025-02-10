#!/bin/bash

# 检查是否提供了目录参数
if [ "$#" -ne 1 ]; then
  echo "用法: $0 <目标目录>"
  exit 1
fi

# 指定要解压文件的目录
TARGET_DIR=$1

# 检查目录是否存在
if [ ! -d "$TARGET_DIR" ]; then
  echo "指定的目录不存在: $TARGET_DIR"
  exit 1
fi

# 切换到目标目录
cd "$TARGET_DIR"

# 解压所有的 .tar.gz 文件到同名目录下
for tar_file in *.tar.gz; do
  if [ -f "$tar_file" ]; then
    # 去掉文件扩展名
    dir_name="${tar_file%.tar.gz}"
    # 创建同名目录
    mkdir -p "$dir_name"
    # 打印正在解压的文件信息
    echo "正在解压: $tar_file 到目录$dir_name"
    # 解压文件到同名目录
    tar -xzf "$tar_file" -C "$dir_name"
  fi
done

# 解压所有的 .zip 文件到同名目录下
for zip_file in *.zip; do
  if [ -f "$zip_file" ]; then
    # 去掉文件扩展名
    dir_name="${zip_file%.zip}"
    # 创建同名目录
    mkdir -p "$dir_name"
    # 打印正在解压的文件信息
    echo "正在解压: $zip_file 到目录$dir_name"
    # 解压文件到同名目录
    unzip "$zip_file" -d "$dir_name"
  fi
done

echo "所有文件解压完成。"
