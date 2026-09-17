"""Turns one Markdown file into the HTML the documentation website shows.

Read README.md in this folder. It explains everything this file does, and why
the mathematics is pulled out before the Markdown parser ever sees it.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

import mistune
from mistune.renderers.html import HTMLRenderer
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name
from pygments.token import STANDARD_TYPES
from pygments.util import ClassNotFound

__all__ = [
    "CODE_PANELS",
    "Heading",
    "RenderedDocument",
    "code_block",
    "contrast_ratio",
    "name_first_heading",
    "pygments_stylesheet",
    "render_markdown",
    "rewrite_link",
    "slugify",
    "strip_markup",
]

# A placeholder that survives Markdown parsing untouched: no punctuation for
# the parser to interpret, and no word an author would type by accident.
_MATH_TOKEN = "zqmathspanzq{}zq"
_MATH_PATTERN = re.compile(r"zqmathspanzq(\d+)zq")

# Fenced code blocks and inline code spans, which must keep their dollar signs.
_PROTECTED = re.compile(
    r"(^```[\s\S]*?^```|^~~~[\s\S]*?^~~~|`[^`\n]+`)", re.MULTILINE
)

# Display mathematics, then inline mathematics. Order matters: the two-dollar
# form has to be taken first or the one-dollar form eats half of it.
_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\$)")

# An assumption or finding identifier mentioned in prose, such as "A-07" or
# "F-11", which the website turns into a link to that entry.
_IDENTIFIER = re.compile(r"\b([AF]-\d{2})\b")

# Anything that is not a letter, a digit, a space or a hyphen. GitHub drops
# these when it builds a heading anchor, and this project's documents contain
# links written against GitHub's result, so the same rule is used here.
_NOT_SLUGGABLE = re.compile(r"[^\w\- ]", re.UNICODE)

_TAG = re.compile(r"<[^>]+>")

_PYGMENTS = HtmlFormatter(nowrap=True)

# The Pygments colour scheme for code in each theme, and the panel and ordinary
# text colour it is drawn with: app.css's --panel and --ink for that theme. A
# test checks these still match app.css, since the colours are chosen against
# them.
_CODE_STYLES = {"light": "default", "dark": "github-dark"}
CODE_PANELS = {
    "light": {"panel": "#f6f7f9", "ink": "#16191d"},
    "dark": {"panel": "#171b21", "ink": "#e6e9ee"},
}

# CLAUDE.md rule 11: ordinary text must reach this contrast against its background.
_MIN_CONTRAST = 4.5

# The first level-one heading of a rendered page, split into its opening tag,
# its words, and the small link to itself that both this site and nbconvert
# put at its end.
_FIRST_HEADING = re.compile(
    r'(<h1\b[^>]*>)(.*?)(<a class="anchor(?:-link)?"[^>]*>[^<]*</a>)?(</h1>)',
    re.DOTALL,
)
# Languages that appear in this project's documents. Anything else is shown
# without colouring rather than guessed at.
_KNOWN_LANGUAGES = frozenset(
    {
        "bash",
        "console",
        "diff",
        "ini",
        "json",
        "pycon",
        "python",
        "sh",
        "shell",
        "text",
        "toml",
        "yaml",
    }
)


@dataclass(frozen=True)
class Heading:
    """One heading of a rendered document, for the on-page contents list."""

    level: int
    text: str
    anchor: str


@dataclass
class RenderedDocument:
    """A Markdown file after rendering: its HTML, its headings and its title."""

    html: str
    title: str
    headings: list[Heading] = field(default_factory=list)
    has_math: bool = False


def strip_markup(text: str) -> str:
    """Remove HTML tags and character escapes, leaving plain words."""
    plain = _TAG.sub("", text)
    plain = plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    plain = plain.replace("&quot;", '"').replace("&#39;", "'")
    return plain.strip()


def slugify(text: str, seen: dict[str, int] | None = None) -> str:
    """Build a heading anchor the way GitHub does, so existing links work."""
    slug = _NOT_SLUGGABLE.sub("", strip_markup(text).lower()).replace(" ", "-")
    slug = slug.strip("-") or "section"
    if seen is None:
        return slug
    count = seen.get(slug, 0)
    seen[slug] = count + 1
    return slug if count == 0 else f"{slug}-{count}"


def _hide_math(source: str) -> tuple[str, list[tuple[bool, str]]]:
    """Replace every formula with a placeholder, and return the formulas."""
    formulas: list[tuple[bool, str]] = []

    def take(match: re.Match[str], display: bool) -> str:
        formulas.append((display, match.group(1).strip()))
        return _MATH_TOKEN.format(len(formulas) - 1)

    pieces: list[str] = []
    for index, piece in enumerate(_PROTECTED.split(source)):
        if index % 2 == 1:  # a code block or a code span; leave it alone
            pieces.append(piece)
            continue
        piece = _DISPLAY_MATH.sub(lambda m: take(m, True), piece)
        piece = _INLINE_MATH.sub(lambda m: take(m, False), piece)
        pieces.append(piece)
    return "".join(pieces), formulas


def _restore_math(rendered: str, formulas: list[tuple[bool, str]]) -> str:
    """Put every formula back, marked up for the browser to typeset."""

    def put(match: re.Match[str]) -> str:
        display, formula = formulas[int(match.group(1))]
        kind = "math-display" if display else "math-inline"
        return f'<span class="{kind}">{html.escape(formula, quote=False)}</span>'

    return _MATH_PATTERN.sub(put, rendered)


def _resolve(target: str, source_path: str) -> str | None:
    """Turn a link relative to one document into a path from the project root."""
    parts = source_path.split("/")[:-1] if "/" in source_path else []
    if target.startswith("/"):
        parts, target = [], target.lstrip("/")
    for piece in target.split("/"):
        if piece in ("", "."):
            continue
        if piece == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(piece)
    return "/".join(parts) if parts else None


def rewrite_link(url: str, source_path: str) -> str:
    """Point a link written for GitHub at the same place on this website."""
    if url.startswith(("http://", "https://", "mailto:", "#", "/api/")):
        return url

    target, _, fragment = url.partition("#")
    if not target:
        return url

    resolved = _resolve(target, source_path)
    if resolved is None:
        return url

    suffix = f"#{fragment}" if fragment else ""
    lowered = resolved.lower()
    if lowered.endswith(".ipynb"):
        return f"#/notebook/{resolved}{suffix}"
    if lowered.endswith(".md"):
        return f"#/doc/{resolved}{suffix}"
    return f"#/source/{resolved}{suffix}"


class _SiteRenderer(HTMLRenderer):
    """Renders one document: anchors the headings and rewrites the links."""

    def __init__(self, source_path: str) -> None:
        super().__init__(escape=False)
        self.source_path = source_path
        self.headings: list[Heading] = []
        self._seen: dict[str, int] = {}

    def heading(self, text: str, level: int, **attrs: object) -> str:
        anchor = slugify(text, self._seen)
        self.headings.append(Heading(level, strip_markup(text), anchor))
        return (
            f'<h{level} id="{anchor}">{text}<a class="anchor" href="#{anchor}"'
            f' aria-label="Link to this section">#</a></h{level}>\n'
        )

    def block_code(self, code: str, info: str | None = None) -> str:
        language = info.strip().split()[0].lower() if info and info.strip() else ""
        return code_block(code, language) + "\n"

    def link(self, text: str, url: str, title: str | None = None) -> str:
        return super().link(text, rewrite_link(url, self.source_path), title)

    def image(self, text: str, url: str, title: str | None = None) -> str:
        return super().image(text, rewrite_link(url, self.source_path), title)


def _link_identifiers(rendered: str) -> str:
    """Turn every "A-07" written in prose into a link to that entry."""
    out: list[str] = []
    # Splitting on tags and code spans keeps identifiers inside an attribute,
    # a heading anchor or a code sample from being linked.
    parts = re.split(r"(<code>[\s\S]*?</code>|<a [\s\S]*?</a>|<[^>]+>)", rendered)
    for index, piece in enumerate(parts):
        if index % 2 == 1:
            out.append(piece)
            continue
        out.append(
            _IDENTIFIER.sub(
                lambda m: f'<a class="identifier" href="#/entry/{m.group(1)}">'
                f"{m.group(1)}</a>",
                piece,
            )
        )
    return "".join(out)


def render_markdown(source: str, *, source_path: str = "") -> RenderedDocument:
    """Render one Markdown document into the HTML the website displays."""
    hidden, formulas = _hide_math(source)
    renderer = _SiteRenderer(source_path)
    parser = mistune.create_markdown(
        renderer=renderer,
        plugins=["table", "strikethrough", "task_lists", "footnotes", "def_list"],
    )
    body = str(parser(hidden))
    body = _restore_math(body, formulas)
    body = _link_identifiers(body)

    title = next(
        (h.text for h in renderer.headings if h.level == 1),
        source_path.rsplit("/", 1)[-1] or "Untitled",
    )
    return RenderedDocument(
        html=body,
        title=title,
        headings=renderer.headings,
        has_math=bool(formulas),
    )


def code_block(code: str, language: str, *, label: bool = True) -> str:
    """One block of code, coloured, with its language named in a corner unless told not to."""
    if language in _KNOWN_LANGUAGES:
        try:
            body = highlight(code, get_lexer_by_name(language), _PYGMENTS)
        except ClassNotFound:  # pragma: no cover - Pygments knows these
            body = html.escape(code)
    else:
        body = html.escape(code)
    badge = (
        f'<span class="code-language">{html.escape(language)}</span>'
        if language and label
        else ""
    )
    return f'<div class="code-block">{badge}<pre><code>{body}</code></pre></div>'


def name_first_heading(rendered: str, name: str) -> str:
    """Show a page's menu name in its first level-one heading, keeping the heading's anchor."""
    return _FIRST_HEADING.sub(
        lambda m: m.group(1) + html.escape(name) + (m.group(3) or "") + m.group(4),
        rendered,
        count=1,
    )


def _channels(colour: str) -> tuple[float, float, float]:
    """The red, green and blue of a colour written #rrggbb, each between 0 and 1."""
    digits = colour.lstrip("#")
    red, green, blue = (int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return red, green, blue


def _luminance(colour: str) -> float:
    """How bright a colour looks, as the contrast formula of the web guidelines defines it."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in _channels(colour)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(first: str, second: str) -> float:
    """The contrast between two colours, from 1 (the same) to 21 (black on white)."""
    brighter, darker = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (brighter + 0.05) / (darker + 0.05)


def _readable(colour: str, panel: str, ink: str) -> str:
    """The colour, mixed toward the ordinary text colour until it reads on the panel."""
    start, target = _channels(colour), _channels(ink)
    for percent in range(0, 101, 5):
        share = percent / 100
        mixed = "#" + "".join(
            f"{round((a + (b - a) * share) * 255):02x}" for a, b in zip(start, target, strict=True)
        )
        if contrast_ratio(mixed, panel) >= _MIN_CONTRAST:
            return mixed
    return ink


def pygments_stylesheet() -> str:
    """The token colours for all code on the site in both themes, each readable on its own panel."""
    # A code block in a document, and a code cell of a notebook, which nbconvert
    # colours with the same token classes.
    scopes = (".code-block pre code", ".notebook-frame .highlight pre")
    rules: list[str] = []
    for theme, style_name in _CODE_STYLES.items():
        style = get_style_by_name(style_name)
        panel, ink = CODE_PANELS[theme]["panel"], CODE_PANELS[theme]["ink"]
        prefix = 'html[data-theme="dark"] ' if theme == "dark" else ""
        # Every token class gets a full rule in both themes, so a colour the
        # light scheme sets can never show through where the dark one sets none.
        for token, name in STANDARD_TYPES.items():
            if not name:
                continue
            found = style.style_for_token(token)
            colour = _readable(f"#{found['color']}", panel, ink) if found["color"] else "inherit"
            weight = "700" if found["bold"] else "400"
            slant = "italic" if found["italic"] else "normal"
            selector = ", ".join(f"{prefix}{scope} .{name}" for scope in scopes)
            rules.append(
                f"{selector} {{ color: {colour}; font-weight: {weight}; font-style: {slant}; }}"
            )
    return "\n".join(rules) + "\n"
