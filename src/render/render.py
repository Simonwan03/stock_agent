from __future__ import annotations

from pathlib import Path
from typing import Dict


def render_markdown(template_dir: str, template_name: str, context: Dict[str, object]) -> str:
    template_path = Path(template_dir) / template_name
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
        for key, value in context.items():
            template = template.replace(f"{{{{ {key} }}}}", str(value))
        return template

    title = context.get("title", "Daily Report")
    summary = context.get("summary", "")
    guidance = context.get("portfolio_guidance", [])
    guidance_lines = "\n".join(f"- {item}" for item in guidance) if guidance else "- None"
    return f"# {title}\n\n{summary}\n\n## Portfolio Guidance\n{guidance_lines}\n"


def write_text(path: str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
