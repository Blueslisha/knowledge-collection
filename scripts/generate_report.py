#!/usr/bin/env python3
"""
知识库报告生成器
模式:
  daily   -> 每日日报
  topic   -> 专题深度报告
"""
import os, sys, json, re
from datetime import datetime
from pathlib import Path

# ---- 从 .env 加载配置 ----
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"[配置] 已加载 {env_path.name}")
except ImportError:
    pass

# ---- 路径 ----
if os.environ.get('GITHUB_WORKSPACE'):
    REPO = Path(os.environ['GITHUB_WORKSPACE'])
else:
    REPO = Path(__file__).resolve().parent.parent

CONFIG = REPO / 'scripts' / 'config.json'

# =============================================
#  LLM 调用（Anthropic / OpenAI 兼容 API）
# =============================================
def llm(system_prompt, user_prompt, test_mode=False):
    if test_mode:
        return "\n## 测试模式 - LLM 摘要\n\n主题: " + user_prompt[:80] + "...\n"

    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')

    if anthropic_key:
        return _llm_anthropic(system_prompt, user_prompt, anthropic_key)
    elif openai_key:
        return _llm_openai(system_prompt, user_prompt, openai_key)
    else:
        return ("\n## [提示] 未配置 API Key\n\n"
                "请设置环境变量:\n"
                "  $env:ANTHROPIC_API_KEY=\"sk-ant-...\"   (Anthropic)\n"
                "  $env:OPENAI_API_KEY=\"sk-...\"           (OpenAI / 兼容 API)\n"
                "  $env:OPENAI_BASE_URL=\"https://...\"      (可选，国内服务商)\n")

MAX_TOKENS = 16384  # 支持超长输出（部分模型可能受限，会自动截断）

def _llm_anthropic(system_prompt, user_prompt, key):
    from anthropic import Anthropic
    res = Anthropic(api_key=key).messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}]
    )
    return res.content[0].text

def _llm_openai(system_prompt, user_prompt, key):
    from openai import OpenAI
    kwargs = {"api_key": key}
    base_url = os.environ.get('OPENAI_BASE_URL', '')
    if base_url:
        kwargs["base_url"] = base_url

    extra = {}
    if os.environ.get('OPENAI_ENABLE_SEARCH', '').lower() in ('1', 'true', 'yes'):
        extra["extra_body"] = {"enable_search": True}

    res = OpenAI(**kwargs).chat.completions.create(
        model=os.environ.get('OPENAI_MODEL', 'gpt-4o'),
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        **extra
    )
    return res.choices[0].message.content

# ---- Slug 生成 ----
def to_slug(text):
    text = re.sub(r'[^\w一-鿿\s-]', '', text.strip().lower())
    return re.sub(r'-+', '-', re.sub(r'\s+', '-', text))[:80] or 'topic'

# ---- 写入文件 ----
def write_file(path, content):
    full = REPO / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding='utf-8')
    print(f"  [OK] {path} ({len(content)} chars)")
    return full

# =============================================
#  模式 1：每日日报
# =============================================
def daily(test_mode=False):
    topics = json.loads(CONFIG.read_text(encoding='utf-8'))['topics']
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    date_cn = f"{now.year} 年 {now.month} 月 {now.day} 日"

    print(f"[日报] {date_cn}  |  专题数: {len(topics)}")

    body = f"# 知识日报 — {date_cn}\n\n---\n\n"

    for i, t in enumerate(topics, 1):
        print(f"  [{i}/{len(topics)}] {t['name']}")
        section = llm(
            "你是一位技术领域的资深研究员。撰写该主题的中文深度研究报告章节。\n\n"
            "格式要求（严格遵循）：\n"
            "1. 章节标题使用「## 一、标题」格式\n"
            "2. 子标题使用「### 1.1 子标题」格式\n"
            "3. 涉及数据必须使用 Markdown 表格对比\n"
            "4. 涉及流程必须使用 ASCII 图表\n"
            "5. 各小节之间用「---」分隔\n"
            "6. 每个章节 800-1500+ 字，越详细越好\n\n"
            "内容要求：\n"
            "- 搜索互联网获取最新信息、数据、动态\n"
            "- 包含核心原理、技术分析、对比评估、趋势展望\n"
            "- 语言专业、客观、精准",
            f"搜索互联网，撰写关于「{t['name']}」的深度研究报告章节。\n"
            f"覆盖：最新进展、核心技术、关键数据、对比分析、趋势展望。",
            test_mode
        )
        body += f"## {i}. {t['name']}\n\n{section}\n\n---\n\n"

    body += f"\n---\n\n*本日报由自动化工作流于 {date_cn} 生成*\n"
    front = (
        f"---\n"
        f"title: 知识日报 — {today}\n"
        f"date: {today}\n"
        f"categories: [daily]\n"
        f"tags: [日报]\n"
        f"---\n\n"
    )
    p = f"daily/{today}.md"
    write_file(p, front + body)
    print(f"  [链接] /{p.replace('.md', '/')}")

# =============================================
#  模式 2：专题研究
# =============================================
def topic(name, test_mode=False):
    today = datetime.now().strftime('%Y-%m-%d')
    slug_name = to_slug(name)

    print(f"[专题] {name}")
    content = llm(
        "你是一位技术领域的资深研究员，正在撰写一份深度研究报告。\n\n"
        "格式要求（严格遵循）：\n"
        "1. 全文标题使用「# 标题」（已存在，正文中不要重复）\n"
        "2. 章节标题使用「## 一、标题」格式\n"
        "3. 子标题使用「### 1.1 标题」或「#### 1.1.1 标题」\n"
        "4. 涉及数据必须使用 Markdown 表格呈现\n"
        "5. 涉及流程必须使用 ASCII 图表（或 mermaid 文本图）\n"
        "6. 各章节之间用「---」分隔\n"
        "7. 文末添加参考资源章节（引用须带链接）\n"
        "8. 全文不低于 6000 字，越多越好\n\n"
        "内容结构（请覆盖）：\n"
        "- 概述 / 背景介绍\n"
        "- 核心技术原理深度剖析\n"
        "- 多方案对比分析（表格）\n"
        "- 应用案例或实践数据\n"
        "- 趋势展望与未来方向\n"
        "- 参考文献（链接 + 标题）\n\n"
        "语言专业、客观、精准，避免空洞套话。",
        f"专题：{name}\n\n"
        f"搜索互联网获取关于「{name}」的最新资料，撰写深度研究报告。",
        test_mode
    )

    front = (
        f"---\n"
        f"title: {name}\n"
        f"date: {today}\n"
        f"categories: [topics]\n"
        f"tags: [{name}]\n"
        f"---\n\n"
        f"# {name}\n\n"
        f"**采集时间**: {today}\n\n"
    )
    p = f"topics/{slug_name}.md"
    write_file(p, front + content)
    print(f"  [链接] /{p.replace('.md', '/')}")

# =============================================
#  模式 3：更新首页索引
# =============================================
def update_index():
    index_path = REPO / 'index.md'
    daily_files = sorted((REPO / 'daily').glob('[0-9]*.md'), reverse=True)
    topic_files = sorted((REPO / 'topics').glob('*.md'))

    sections = []
    old = index_path.read_text(encoding='utf-8') if index_path.exists() else ''
    hero = re.search(r'(<section class="hero">.*?</section>)', old, re.DOTALL)
    if hero:
        sections.append(hero.group(1))
    else:
        sections.append('<section class="hero"><h1 class="hero-title">知识集</h1></section>')

    sections.append("""<section class=\"section\"><div class=\"section-header\"><span class=\"section-line\"></span><h2 class=\"section-title\">最近日报</h2></div><div class=\"post-list\">\n""")
    for f in daily_files[:10]:
        n = f.stem
        sections.append(f"""    <article class=\"post-item\"><time class=\"post-item-date\">{n[:4]}.{n[5:7]}.{n[8:10]}</time><h3 class=\"post-item-title\"><a href=\"/daily/{n}/\">知识日报 - {n}</a></h3></article>\n""")
    sections.append("""</div><p class=\"section-footnote\"><a href=\"/daily/\">查看全部 -></a></p></section>\n""")

    sections.append("""<section class=\"section\"><div class=\"section-header\"><span class=\"section-line\"></span><h2 class=\"section-title\">专题</h2></div><div class=\"topic-grid\">\n""")
    for f in topic_files:
        sn = to_slug(f.stem)
        sections.append(f"""    <a href=\"/topics/{sn}/\" class=\"topic-item\"><span class=\"topic-icon\">--</span><span class=\"topic-name\">{f.stem}</span></a>\n""")
    sections.append("</div></section>")

    index_path.write_text('\n'.join(sections), encoding='utf-8')
    print(f"  [OK] 首页索引已更新")

# =============================================
#  主入口
# =============================================
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法:\n  python scripts/generate_report.py daily\n  python scripts/generate_report.py topic '主题'\n  python scripts/generate_report.py test")
        sys.exit(1)

    m = sys.argv[1]
    if m == 'daily':
        daily()
    elif m == 'topic':
        if len(sys.argv) < 3:
            print("[错误] 请指定专题名称")
            sys.exit(1)
        topic(sys.argv[2])
    elif m == 'test':
        print("[测试模式]\n")
        daily(test_mode=True)
        print()
        topic("测试专题", test_mode=True)
        print("\n[完成]")
    else:
        print(f"[错误] 未知模式: {m}")
