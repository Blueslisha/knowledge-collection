---
layout: default
title: 首页
---

<section class="hero">
  <h1 class="hero-title">知识集</h1>
  <p class="hero-subtitle">个人知识库 · 自动采集 · 每日更新</p>
  <p class="hero-desc">由 AI 智能体自动检索、整理、归档的信息集合。涵盖前沿技术、学术动态与深度专题。</p>
  <div class="hero-meta">
    <span class="meta-badge">持续更新中</span>
    <span class="meta-badge">14 篇条目</span>
  </div>
</section>

<section class="section">
  <h2 class="section-title">
    <span class="section-line"></span>
    最近日报
  </h2>
  <div class="post-list">
    <article class="post-card" style="--i: 0">
      <time class="post-date">2026-05-15</time>
      <h3 class="post-title"><a href="{{ '/daily/2026-05-15/' | relative_url }}">知识日报 — 2026-05-15</a></h3>
      <p class="post-excerpt">RAG 技术前沿 · AI Agent 框架对比 · 本周值得关注的论文</p>
    </article>
    <p class="section-footnote">每日 08:00 自动更新 · <a href="{{ '/daily/' | relative_url }}">查看全部 →</a></p>
  </div>
</section>

<section class="section">
  <h2 class="section-title">
    <span class="section-line"></span>
    专题文章
  </h2>
  <div class="topic-grid">
    <a href="{{ '/topics/llm-architecture/' | relative_url }}" class="topic-card" style="--i: 0">
      <span class="topic-icon">▴</span>
      <span class="topic-name">LLM 架构深度解析</span>
      <span class="topic-count">1 篇文章</span>
    </a>
  </div>
</section>

<section class="section section--about">
  <h2 class="section-title">
    <span class="section-line"></span>
    关于本库
  </h2>
  <div class="about-content">
    <p>本知识库由 <strong>Coze 智能体「知识库管家」</strong> 自动维护。每日定时检索指定领域的最新信息，经 AI 归纳总结后以 Markdown 格式存档于此，并通过 GitHub Pages 渲染为可阅读的 HTML 页面。</p>
    <div class="about-features">
      <div class="feature">
        <span class="feature-icon">◈</span>
        <span>定时采集 — 每日 08:00 自动检索</span>
      </div>
      <div class="feature">
        <span class="feature-icon">◈</span>
        <span>主动查询 — 随时对话触发深度搜索</span>
      </div>
      <div class="feature">
        <span class="feature-icon">◈</span>
        <span>结构化归档 — Markdown 格式，版本化管理</span>
      </div>
    </div>
  </div>
</section>

<footer class="site-footer">
  <p>由 Coze 智能体驱动 · 托管于 GitHub Pages</p>
</footer>
