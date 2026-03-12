from __future__ import annotations

import csv
import io
import json
import shlex
from pathlib import Path
from typing import Any

import pandas as pd


SMILES_ALIASES = {"smiles", "canonical_smiles"}
ID_ALIASES = {"id", "compound_id", "zinc_id"}
NAME_ALIASES = {"name", "compound_name"}


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _norm_col(c: str) -> str:
    return c.strip().lower()


def parse_library(
    input_path: str,
    input_format: str,
    smi_layout: str = "smiles_id",
    header: str = "auto",
    comment_prefix: str = "#",
    allow_cpp_comments: bool = True,
    name_is_rest: bool = True,
    smi_quoted_name: bool = False,
    sep: str = ",",
    quotechar: str = '"',
    smiles_col: str | None = None,
    id_col: str | None = None,
    name_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fmt = input_format.lower()
    if fmt == "smi":
        return parse_smi(
            input_path=input_path,
            smi_layout=smi_layout,
            header=header,
            comment_prefix=comment_prefix,
            allow_cpp_comments=allow_cpp_comments,
            name_is_rest=name_is_rest,
            smi_quoted_name=smi_quoted_name,
        )
    if fmt == "csv":
        return parse_csv(
            input_path=input_path,
            header=header,
            sep=sep,
            quotechar=quotechar,
            smiles_col=smiles_col,
            id_col=id_col,
            name_col=name_col,
        )
    raise ValueError(f"Unsupported input_format={input_format}")


def parse_smi(
    input_path: str,
    smi_layout: str,
    header: str,
    comment_prefix: str,
    allow_cpp_comments: bool,
    name_is_rest: bool,
    smi_quoted_name: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token_counts: dict[int, int] = {}
    skipped_blank = 0
    skipped_comment = 0
    header_detected = False
    header_tokens: list[str] = []

    text = Path(input_path).read_text(encoding="utf-8-sig", errors="replace")
    for idx, raw in enumerate(io.StringIO(text)):
        line = raw.rstrip("\n")
        stripped = line.strip()
        if not stripped:
            skipped_blank += 1
            continue
        if stripped.startswith(comment_prefix) or (allow_cpp_comments and stripped.startswith("//")):
            skipped_comment += 1
            continue

        if header == "auto" and not header_detected and not rows:
            tks = stripped.split()
            low = {_norm_col(t) for t in tks}
            if low & (SMILES_ALIASES | ID_ALIASES | NAME_ALIASES):
                header_detected = True
                header_tokens = tks
                continue
        if _to_bool(header) and not header_detected and not rows:
            header_detected = True
            header_tokens = stripped.split()
            continue

        tokens = shlex.split(stripped) if smi_quoted_name else stripped.split()
        token_counts[len(tokens)] = token_counts.get(len(tokens), 0) + 1
        rec = {
            "source_row": idx,
            "compound_id": None,
            "compound_name": None,
            "smiles": None,
            "raw_smiles": tokens[0] if tokens else "",
            "raw_id": None,
            "raw_name": None,
            "raw_line": line[:1000],
            "parse_status": "ok",
            "parse_error": None,
            "extra_tokens_count": max(0, len(tokens) - 1),
        }
        if not tokens:
            rec["parse_status"] = "fail"; rec["parse_error"] = "empty_line"
            rows.append(rec); continue
        rec["smiles"] = tokens[0]
        if smi_layout == "smiles_only":
            rec["compound_id"] = f"row_{idx}"
        elif smi_layout == "smiles_id":
            if len(tokens) < 2:
                rec["parse_status"] = "fail"; rec["parse_error"] = "too_few_columns"
            else:
                rec["compound_id"] = str(tokens[1]); rec["raw_id"] = str(tokens[1])
        elif smi_layout == "smiles_name_id":
            if len(tokens) < 3:
                rec["parse_status"] = "fail"; rec["parse_error"] = "too_few_columns"
            else:
                if name_is_rest:
                    rec["compound_name"] = " ".join(tokens[1:-1]).strip()
                    rec["compound_id"] = str(tokens[-1])
                else:
                    rec["compound_name"] = str(tokens[1]); rec["compound_id"] = str(tokens[2])
                rec["raw_name"] = rec["compound_name"]; rec["raw_id"] = rec["compound_id"]
        else:
            raise ValueError(f"Unsupported smi_layout={smi_layout}")

        if not rec["smiles"]:
            rec["parse_status"] = "fail"; rec["parse_error"] = "missing_smiles"
        rows.append(rec)

    df = pd.DataFrame(rows)
    manifest = {
        "format": "smi",
        "path": str(input_path),
        "header_detected": header_detected,
        "header_tokens": "|".join(header_tokens),
        "skipped_blank": skipped_blank,
        "skipped_comment": skipped_comment,
        "token_count_distribution": json.dumps(token_counts, sort_keys=True),
    }
    return df, manifest


def parse_csv(
    input_path: str,
    header: str,
    sep: str,
    quotechar: str,
    smiles_col: str | None,
    id_col: str | None,
    name_col: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    header_row = 0 if _to_bool(header) or header == "auto" else None
    pdf = pd.read_csv(
        input_path,
        sep=sep,
        quotechar=quotechar,
        header=header_row,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    if header_row is None:
        pdf.columns = [f"col_{i}" for i in range(pdf.shape[1])]

    nmap = {_norm_col(c): c for c in pdf.columns}

    def resolve(col: str | None, aliases: set[str], required: bool) -> str | None:
        if col:
            if col in pdf.columns:
                return col
            low = _norm_col(col)
            if low in nmap:
                return nmap[low]
            raise ValueError(f"Column '{col}' not found in CSV")
        for a in aliases:
            if a in nmap:
                return nmap[a]
        if required:
            raise ValueError(f"Required column not found. aliases={sorted(aliases)}")
        return None

    smiles_c = resolve(smiles_col, SMILES_ALIASES, True)
    id_c = resolve(id_col, ID_ALIASES, False)
    name_c = resolve(name_col, NAME_ALIASES, False)

    rows = []
    for i, row in pdf.iterrows():
        source_row = i + (1 if header_row == 0 else 0)
        smi = str(row.get(smiles_c, "")).strip()
        cid = str(row.get(id_c, "")).strip() if id_c else ""
        if not cid:
            cid = f"row_{source_row}"
        cname = str(row.get(name_c, "")).strip() if name_c else ""
        rec = {
            "source_row": int(source_row),
            "compound_id": cid,
            "compound_name": cname or None,
            "smiles": smi,
            "raw_smiles": smi,
            "raw_id": cid,
            "raw_name": cname or None,
            "raw_row_json": row.to_json(force_ascii=False),
            "parse_status": "ok",
            "parse_error": None,
        }
        if not smi:
            rec["parse_status"] = "fail"
            rec["parse_error"] = "missing_smiles"
        rows.append(rec)

    manifest = {
        "format": "csv",
        "path": str(input_path),
        "header_detected": bool(header_row == 0),
        "smiles_col": smiles_c,
        "id_col": id_c,
        "name_col": name_c,
        "n_columns": int(len(pdf.columns)),
    }
    return pd.DataFrame(rows), manifest
