---
title: "Notion + RSS - 构建自己的稍后读"
date: "2025-06-28T21:15:09+08:00"
lastmod: "2025-06-28T21:15:09+08:00"
draft: false
description: "用 Notion 数据库和 RSS 技术打造属于自己的稍后读系统，结合自动化脚本与 Caddy 部署，轻松实现多平台内容聚合与高效信息管理。"
tags: ["notion", "RSS", "caddy", "infomation"]
series: "make feed"
---

本篇介绍如何利用 Notion 数据库与 RSS 技术，构建属于自己的稍后读系统。通过自动化脚本将 Notion 中的内容转为 RSS 订阅源，并结合 Caddy 部署，实现多平台内容聚合、自动推送与高效信息管理。适合希望提升信息流转效率、打造个人知识管理闭环的用户。

<!--more-->

项目地址: [GitHub - make_feed](https://github.com/luke396/make_feed)

## 缘起

我使用 FreshRSS 进行关注信息管理已经有些时间，虽然大部分时候还是忍不住继续刷刷刷，沉迷在数据流的瀑布中。大部分应用都热衷于淡化传统的订阅方式，包括但不限于不直接推送内容更新，转而使用或许更有助于吸收用户时间的推荐系统。推荐 vs 订阅的问题，不是这里讨论的重点。我习惯利用阅读获取严肃信息，想要长期利用电脑追踪各个平台用户的内容，RSS 是我目前找到的最合适的工具。

目前我的主要工具是自部署的 RSSHub 和 FreshRSS 作为最主要的订阅链接生成和阅读平台，也是现在主流可以免费使用的工具。由于 Twitter 的不断封闭，RSSHub + Twitter token 的方式可用性也逐渐降低，故使用 Folo 进行补充。可以猜想，Folo 要么是构建了大量的账号池进行内容抓取，要么是将别的用户通过各种方式获取到的推文内容储存服务器中再进行复用，要么二者兼有。

稍后读是个经典的 " 伪需求 "，不断地有软件推出各种方案，达人给出各样流程，我最初使用简悦的网页裁剪再保存到本地，存了一大堆网页稍后要读，自然，稍后从未到来。随着使用经验的不断加深，我的 " 第二大脑 " - 目前由 obsidian 进行扮演，只保留我阅读过并且觉得有意义进行长久化保存的内容。RSS 的订阅库，就充当了搜集材料的主要方式，RSS 阅读器自然就是第一次过滤的关键工具。

很多零碎的网页、文件没有办法也没有必要通过直接地 RSS 链接进行订阅，利用 Notion 数据库充当在线稍后读的储存库，再利用代码，获取更新并生成 RSS 文件，给 FreshRSS 进行订阅和阅读。如果有需要，再保存到本地的 obsidian 中。

## 构建

这个思路的产生和实现都由 claude-sonnet-4 大力支持，具体的代码文件就不再进行解读。详解代码总会因为代码的更新而失效，也让博客变成了那个代码仓库的技术文档，虽然没有技术，也没有文档。

总的思路就是，首先发现内容，然后利用 Notion 的 web clipper 保存到指定的 Database 中。紧接着，创建 Notion 的 token，并找到对应 Database 的 id，可以利用 `.env` 进行保存和读取。接着把 Database 的更新到本地，通过解析 API 的返回并构建 xml 订阅文件，就把 Notion 的 Database 变成可以由 RSS 阅读器进行解析的合适内容载体。最后，只需要将生成的文件放到指定位置，并暴露给 RSS 阅读器就好，我是用的是 docker 部署的 caddy，只要映射进入 site，然后配置相应的 Caddyfile，生成定期任务更新就可以了。

下面是没有在 Github 写出的 caddy 的相关配置

```
volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
      - ./feeds:/srv/feeds # 映射对应xml文件夹
```

```
# 暴露到8080端口，后续通过类似 http://caddy:8080/you-feed.xml 进行访问/添加订阅即可
:8080 {
    root * /srv/feeds
    file_server

    @xml path *.xml
    header @xml Content-Type "application/xml; charset=utf-8"
}
```
