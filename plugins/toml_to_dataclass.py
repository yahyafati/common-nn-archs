from __future__ import annotations

import sys
import tomllib
from typing import Any, Dict, Tuple


def infer_type(key: str, value: Any) -> str:
    """Basic type inference (extend as needed)."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    return "str"


def to_class_name(name: str) -> str:
    """Convert toml section name → ClassName."""
    return "".join(part.capitalize() for part in name.split("_")) + "Config"


def parse_toml_structure(data: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Flatten TOML into:
    - root fields
    - nested section configs
    """
    root_fields = {}
    sections = {}

    for key, value in data.items():
        if isinstance(value, dict):
            sections[key] = value
        else:
            root_fields[key] = value

    return root_fields, sections


def generate_dataclass(name: str, fields: Dict[str, Any], is_root: bool = False) -> str:
    lines = [f"@dataclass(frozen=True)", f"class {name}:"]
    if not fields:
        lines.append("    pass")
        return "\n".join(lines)

    for k, v in fields.items():
        if not is_root:
            lines.append(f"    {k}: {infer_type(k,v)}")
        else:
            lines.append(f"    {k}: {v}")

    return "\n".join(lines)


def generate_schema(toml_data: Dict[str, Any]) -> str:
    root_fields, sections = parse_toml_structure(toml_data)

    output = ["from dataclasses import dataclass\n"]
    fields = {k: infer_type(k, v) for k, v in root_fields.items()}
    config_sections = {k: f'"{to_class_name(k)}"' for k in sections.keys()}
    fields.update(config_sections)
    output.append(
        generate_dataclass(
            "Config",
            fields,
            True,
        )
    )
    output.append("")

    for name, content in sections.items():
        class_name = to_class_name(name)
        output.append(generate_dataclass(class_name, content))
        output.append("")

    return "\n".join(output)


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    with open(input_path, "rb") as f:
        raw_data = tomllib.load(f)

    schema = generate_schema(raw_data)
    with open(output_path, "w") as f:
        f.write(schema)
