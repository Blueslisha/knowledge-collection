---
layout: default
title: 首页
---

<section class="hero">
  <h1 class="hero-title">知识集</h1>
  <p class="hero-sublabel">— personal knowledge archive —</p>
  <div class="hero-meta-row">
    <span class="hero-meta">每日更新 · 持续归档</span>
  </div>
  <div class="hero-divider"></div>
  <p class="hero-desc">由 AI 智能体自动检索、整理、归档的信息集合。涵盖前沿技术、学术动态与深度专题。</p>
  <div class="hero-badge-row">
    <span class="meta-badge">14 篇条目</span>
    <span class="meta-badge">2 个专题</span>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <span class="section-line"></span>
    <h2 class="section-title">最近日报</h2>
  </div>

  <div class="post-list">
    <article class="post-item">
      <time class="post-item-date">2026.05.15</time>
      <h3 class="post-item-title"><a href="{{ '/daily/2026-05-15/' | relative_url }}">知识日报 — RAG 技术前沿 · AI Agent 框架对比</a></h3>
      <p class="post-item-excerpt">混合检索策略、轻量化重排序、LangGraph vs CrewAI</p>
    </article>
  </div>

  <p class="section-footnote"><a href="{{ '/daily/' | relative_url }}">查看全部日报 →</a></p>
</section>

<section class="section">
  <div class="section-header">
    <span class="section-line"></span>
    <h2 class="section-title">专题</h2>
  </div>

  <div class="topic-grid">
    <a href="{{ '/topics/llm-architecture/' | relative_url }}" class="topic-item">
      <span class="topic-icon">◆</span>
      <span class="topic-name">LLM 架构深度解析</span>
      <span class="topic-count">1 篇</span>
    </a>
    <span class="topic-item" style="opacity: 0.35; cursor: default;">
      <span class="topic-icon">◇</span>
      <span class="topic-name">RAG 实践指南</span>
      <span class="topic-count">即将推出</span>
    </span>
  </div>
</section>

<section class="section">
  <div class="section-header">
    <span class="section-line"></span>
    <h2 class="section-title">关于</h2>
  </div>

  <div class="about-content">
    <p>本知识库由 Coze 智能体「<strong>知识库管家</strong>」自动维护。每日定时检索指定领域的最新信息，经 AI 归纳总结后以 Markdown 格式存档于此，并通过 GitHub Pages 渲染为可阅读的 HTML 页面。</p>
    <div class="about-features">
      <div class="feature">
        <span class="feature-icon">◆</span>
        <span>定时采集 — 每日 08:00 自动检索</span>
      </div>
      <div class="feature">
        <span class="feature-icon">◆</span>
        <span>主动查询 — 随时对话触发深度搜索</span>
      </div>
      <div class="feature">
        <span class="feature-icon">◆</span>
        <span>结构化归档 — Markdown 格式，版本化管理</span>
      </div>
    </div>
  </div>
</section>

<footer class="site-footer">
  <p>Coze 智能体驱动 · GitHub Pages 托管</p>
</footer>
