"""Render runtime-assigned docstrings as structured Quarto pages."""

from __future__ import annotations

import inspect
from pathlib import Path

import astroapers


API_DIR = Path(__file__).resolve().parents[1] / "api"
RUNTIME_PREFIXES = ("apsum_", "npix_")


def main() -> None:
    for name in astroapers.__all__:
        if not name.startswith(RUNTIME_PREFIXES):
            continue
        obj = getattr(astroapers, name)
        if not inspect.isfunction(obj) or not obj.__doc__:
            continue
        path = API_DIR / f"{name}.qmd"
        if path.exists():
            path.write_text(_render_function_page(name, obj), encoding="utf-8")


def _render_function_page(name: str, obj) -> str:
    parsed = _parse_numpy_doc(obj.__doc__)
    out = [
        f"# {name} {{ #astroapers.{name} }}",
        "",
        "```python",
        _signature(name, obj),
        "```",
        "",
    ]
    if parsed["summary"]:
        out.extend([parsed["summary"], ""])
    if parsed["parameters"]:
        out.extend(_parameter_table(parsed["parameters"], obj))
    if parsed["returns"]:
        out.extend(_returns_table(parsed["returns"]))
    if parsed["notes"]:
        out.extend(["## Notes {.doc-section .doc-section-notes}", ""])
        out.extend([parsed["notes"], ""])
    return "\n".join(out).rstrip() + "\n"


def _parameter_table(rows, obj) -> list[str]:
    defaults = _signature_defaults(obj)
    out = [
        "## Parameters {.doc-section .doc-section-parameters}",
        "",
    ]
    for row in rows:
        default = ""
        if "," not in row["name"]:
            default = defaults.get(row["name"], "")
        out.extend(_definition_item(row["name"], row["type"], row["description"], default))
    return out


def _returns_table(rows) -> list[str]:
    out = [
        "## Returns {.doc-section .doc-section-returns}",
        "",
    ]
    for row in rows:
        out.extend(_definition_item(row["name"], row["type"], row["description"]))
    return out


def _parse_numpy_doc(doc: str) -> dict[str, object]:
    lines = inspect.cleandoc(doc).splitlines()
    sections: dict[str, list[str]] = {}
    summary: list[str] = []
    idx = 0
    current = summary
    while idx < len(lines):
        if idx + 1 < len(lines) and lines[idx + 1].strip().startswith("---"):
            title = lines[idx].strip().lower()
            current = []
            sections[title] = current
            idx += 2
            continue
        current.append(lines[idx])
        idx += 1

    return {
        "summary": _clean_text(summary),
        "parameters": _parse_fields(sections.get("parameters", [])),
        "returns": _parse_fields(sections.get("returns", [])),
        "notes": _clean_text(sections.get("notes", [])),
    }


def _parse_fields(lines: list[str]) -> list[dict[str, str]]:
    fields = []
    current: dict[str, object] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith(" ") and " : " in line:
            if current is not None:
                fields.append(_finish_field(current))
            name, type_ = stripped.split(" : ", 1)
            current = {"name": name, "type": type_, "description": []}
            continue
        if current is not None:
            current["description"].append(stripped)
    if current is not None:
        fields.append(_finish_field(current))
    return fields


def _finish_field(field: dict[str, object]) -> dict[str, str]:
    return {
        "name": str(field["name"]),
        "type": str(field["type"]),
        "description": _clean_text(field["description"]),
    }


def _clean_text(lines) -> str:
    return " ".join(str(line).strip() for line in lines if str(line).strip())


def _signature_defaults(obj) -> dict[str, str]:
    defaults = {}
    for name, param in inspect.signature(obj).parameters.items():
        if param.default is not inspect.Parameter.empty:
            defaults[name] = repr(param.default)
    return defaults


def _signature(name: str, obj) -> str:
    params = []
    inserted_kw_marker = False
    for param in inspect.signature(obj).parameters.values():
        if param.kind is inspect.Parameter.KEYWORD_ONLY and not inserted_kw_marker:
            params.append("*")
            inserted_kw_marker = True
        text = param.name
        if param.default is not inspect.Parameter.empty:
            text += f"={param.default!r}"
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            text = "*" + text
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            text = "**" + text
        params.append(text)

    flat = f"{name}({', '.join(params)})"
    if len(flat) <= 80:
        return flat
    lines = [f"{name}("]
    lines.extend(f"    {param}," for param in params)
    lines.append(")")
    return "\n".join(lines)


def _definition_item(
    name: str, type_: str, description: str, default: str = ""
) -> list[str]:
    annotation = f" [:]{{.parameter-annotation-sep}} [{type_}]{{.parameter-annotation}}"
    default_text = (
        f" [ = ]{{.parameter-default-sep}} [{default}]{{.parameter-default}}"
        if default
        else ""
    )
    return [
        f"<code>[**{_text(name)}**]{{.parameter-name}}{annotation}{default_text}</code>",
        "",
        f":   {_text(description)}",
        "",
    ]


def _text(value) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text


if __name__ == "__main__":
    main()
