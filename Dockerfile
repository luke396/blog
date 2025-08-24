FROM hugomods/hugo:exts

# 定义域名构建参数
ARG DOMAIN=lukewzb.site

# 安装 git（移除 SSH 客户端）
RUN apk add --no-cache git

# 设置工作目录
WORKDIR /src

# 克隆公开仓库（无需 SSH 密钥）
RUN git clone https://github.com/luke396/blog.git . && \
    git submodule update --init --recursive

# 设置域名环境变量
ENV BLOG_DOMAIN=${DOMAIN}

# 暴露端口
EXPOSE 1313

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:1313/ || exit 1

# 启动 Hugo 服务器，使用环境变量
CMD hugo server --bind=0.0.0.0 --port=1313 --appendPort=false --baseURL=https://${BLOG_DOMAIN} --enableGitInfo --disableFastRender