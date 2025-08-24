#!/bin/bash

# Hugo Blog 更新脚本
# 用于从 GitHub 拉取最新内容并重新部署

set -e  # 遇到错误立即退出

echo "开始更新 Hugo 博客..."

# 检查 .env 文件是否存在
if [ ! -f .env ]; then
    echo "错误: .env 文件不存在！"
    echo "请复制 .env.example 为 .env 并配置域名"
    exit 1
fi

# 显示当前域名配置
DOMAIN=$(grep BLOG_DOMAIN .env | cut -d'=' -f2)
echo "当前配置域名: $DOMAIN"

# 拉取最新的基础镜像
echo "拉取最新 Hugo 基础镜像..."
docker pull hugomods/hugo:exts

# 重新构建博客镜像（不使用缓存以获取最新内容）
echo "重新构建博客镜像..."
docker-compose build --no-cache

# 重新部署服务
echo "重新部署服务..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
echo "检查服务状态..."
if docker-compose ps | grep -q "Up"; then
    echo "博客更新成功！"
    echo "访问地址: https://$DOMAIN"
    
    # 显示容器状态
    docker-compose ps
    
    # 显示最近的日志
    echo ""
    echo "最近的日志:"
    docker-compose logs --tail=10 hugo
else
    echo "服务启动失败，请检查日志:"
    docker-compose logs hugo
    exit 1
fi