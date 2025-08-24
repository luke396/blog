# Zhengbo Wang's Blog

基于 Hugo 和 PaperMod 主题的个人技术博客。

## 本地开发

```bash
git clone https://github.com/luke396/blog.git
cd blog
git submodule update --init --recursive
hugo server --enableGitInfo
```

访问 <http://localhost:1313>

## 部署

```bash
cp .env.example .env
# 编辑 .env 设置域名
docker-compose up -d --build
```

更新内容：

```bash
./update.sh
```
