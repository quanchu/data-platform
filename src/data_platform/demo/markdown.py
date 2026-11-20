"""Render untrusted Markdown in an isolated preview without active HTML."""

from html import escape

from markdown_it import MarkdownIt


def render_preview(source: str) -> str:
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    # Preview must not fetch remote image URLs when an article is opened.
    parser.renderer.rules["image"] = lambda tokens, idx, options, env: escape(
        tokens[idx].content
    )
    body = parser.render(source)
    return (
        """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<style>body{font:16px/1.65 system-ui,sans-serif;color:#20392e;padding:12px 24px;max-width:900px;margin:auto;overflow-wrap:anywhere}h1,h2,h3{line-height:1.25}h1{font-size:28px}h2{font-size:22px;margin-top:1.8em}pre{overflow:auto;padding:16px;background:#f0f5f2}blockquote{border-left:3px solid #187962;margin-left:0;padding-left:20px}table{border-collapse:collapse}td,th{border:1px solid #ccd8d0;padding:8px}a{color:#187962}</style></head><body>"""
        + body
        + "</body></html>"
    )
