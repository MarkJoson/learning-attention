"""notebook 呈现工具：中文字体 + 暗色代码高亮（适配深色编辑器）。

两个常见 notebook 痛点：
  1. matplotlib 默认字体（DejaVu Sans）不含中文字形 → 中文标题/图例显示成方框；
  2. IPython 的 `Code(...)` 用 pygments 固定**亮色** style 生成内联 HTML，黑背景编辑器下刺眼，
     且静态 HTML 无法跟随查看器主题。

`setup_cjk()` 解决 1；`show_code()` 用**暗色** pygments style 解决 2（匹配深色编辑器）。
"""
from __future__ import annotations

# 优先级从高到低；取系统实际存在者。Noto Sans SC / SimHei 在多数 Linux + 本机均可用。
_CJK_FONTS = ["Noto Sans SC", "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
              "Source Han Sans SC", "Droid Sans Fallback"]


def setup_cjk(fonts: list[str] | None = None) -> str:
    """让 matplotlib 正常显示中文（否则中文是方框）。返回实际选用的首选字体名。"""
    import matplotlib
    from matplotlib.font_manager import fontManager

    available = {f.name for f in fontManager.ttflist}
    picked = [f for f in (fonts or _CJK_FONTS) if f in available]
    # 选中的中文字体放最前，保留原有 sans-serif 作英文/fallback
    matplotlib.rcParams["font.sans-serif"] = picked + list(matplotlib.rcParams["font.sans-serif"])
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免缺字
    return picked[0] if picked else "(未找到中文字体)"


def show_code(path, language: str = "python", style: str = "one-dark", max_height: int = 600):
    """用**暗色** pygments style 显示源码，适配深色编辑器（IPython 默认 `Code` 是白底）。

    style 默认 ``one-dark``（深背景）；亮色编辑器可传 ``style='default'`` / ``'github-light'`` 等。
    注：notebook 输出是静态 HTML，无法动态跟随查看器主题；这里固定用暗色匹配深色编辑器。
    """
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    from IPython.display import HTML

    with open(path, encoding="utf-8") as f:
        code = f.read()
    inner = highlight(code, get_lexer_by_name(language), HtmlFormatter(style=style, noclasses=True))
    return HTML(
        f'<div style="max-height:{max_height}px;overflow:auto;border-radius:6px;font-size:90%">{inner}</div>'
    )
