+++
date = '2025-02-04T20:54:17+08:00'
draft = false
title = 'Chatdoc Day1'
+++

> 本来的计划，是与类似 OpenWebUI 等网页对话 + Copilot 进行合作，通过网页对话制定计划，再阅读框架和文档去实现具体代码。

> 现在可能与计划有些出入，因为 [Cline](https://github.com/cline/cline) 的出现，虽然我仍然对于让 LLM coding agent 完全写出代码抱有怀疑，但从它写的 demo 入手，或许是个更 "AI 时代 " 的选择

## Cline

首先是 Cline 的设置，由于 Deepseek 无论是硅基流动的，还是官方的 API 都不是很稳定，可以利用 Copilot 的 claude-3-5。

> 这里的担忧是，会不会过分地请求 Copilot 导致被 GitHub 警告，因为这个问题曾经在 [avanate-nvim](https://github.com/yetone/avante.nvim) 中出现。我记不清在具体什么位置，看到了一个大概 1M/hour 的限制。

接着稍微读了一下 Cline 的文档，觉得最重要的是设置 [memory bank](https://docs.cline.bot/improving-your-prompting-skills/custom-instructions-library/cline-memory-bank)，就是利用 `cline_docs/` 下的一系列文件和 [自定义的 prompt](https://docs.cline.bot/improving-your-prompting-skills/custom-instructions-library/cline-memory-bank#id-4.-custom-instructions) 让 Cline 可以跟踪项目的进度，并更新。

> 我把这个 memory 排除在了 git 之外，总归还是有点私密的

**有一点疑问就是，这些 memory 的最佳实践，是否需要手动更改，还是更推荐让 Cline 全程自助处理。**

另一个，应该可能有所发挥，但我目前还没找到合适场景的，是 `.clinerules`，类似 cursor 的配置文件，也提供了 [模板](https://docs.cline.bot/getting-started/getting-started-new-coders/our-favorite-tech-stack#clinerules-template)，可能更具体的实践用法，还需要未来探索。

## 项目详情

已经生成了 memory 和初步的框架，具体写在明天。

> 虽然是懒惰的程度居多，但强行辩解就是，为了明天开始写 Day2 留一个开头，让这个工作流滚动起来。
