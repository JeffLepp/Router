from __future__ import annotations

import re


_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_DML = re.compile(r"\b(?:insert|update|delete|merge|drop|alter|truncate)\b", re.I)


def solve(prompt: str) -> str | None:
    """Return only SQL whose requested structure can be reconstructed mechanically."""

    if _DML.search(prompt):
        return None
    lower = prompt.lower()
    if "sql" not in lower and "table" not in lower:
        return None
    for handler in (_select_equals_literal, _repair_above_average, _create_table):
        answer = handler(prompt)
        if answer is not None:
            return answer
    return None


def _select_equals_literal(prompt: str) -> str | None:
    if not re.search(r"\bselect\s+all\s+columns\b", prompt, re.I):
        return None
    table_hits = re.findall(
        rf"\btable\s+(?:named\s+)?`?({_IDENT})`?",
        prompt,
        re.I,
    )
    where = re.search(
        rf"\bwhere\s+(?:the\s+)?`?({_IDENT})`?\s+(?:column\s+)?(?:equals?|=)\s+"
        r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|[-+]?\d+(?:\.\d+)?)",
        prompt,
        re.I,
    )
    if len(table_hits) != 1 or not where:
        return None
    table = table_hits[0]
    column, literal = where.groups()
    if not _safe_identifiers(table, column) or prompt[where.end() :].count("="):
        return None
    if literal.startswith('"'):
        literal = "'" + literal[1:-1].replace("''", "'").replace("'", "''") + "'"
    return f"SELECT * FROM {table} WHERE {column} = {literal};"


def _repair_above_average(prompt: str) -> str | None:
    lower = prompt.lower()
    if not re.search(r"\b(?:above|greater than|higher than)\b.{0,35}\baverage\b", lower, re.S):
        return None
    blocks = re.findall(r"```sql\s*(.*?)```", prompt, re.I | re.S)
    if len(blocks) != 1:
        return None
    sql = " ".join(blocks[0].split()).rstrip(";")
    match = re.fullmatch(
        rf"SELECT\s+({_IDENT}(?:\s*,\s*{_IDENT})*)\s+FROM\s+({_IDENT})\s+"
        rf"WHERE\s+({_IDENT})\s*>\s*(?:[-+]?\d+(?:\.\d+)?|AVG\s*\(\s*({_IDENT})\s*\))",
        sql,
        re.I,
    )
    if not match:
        return None
    select_raw, table, column, avg_column = match.groups()
    selected = [part.strip() for part in select_raw.split(",")]
    if not _safe_identifiers(table, column, *selected):
        return None
    if avg_column and avg_column.lower() != column.lower():
        return None
    select_clause = ", ".join(selected)
    return (
        f"SELECT {select_clause}\n"
        f"FROM {table}\n"
        f"WHERE {column} > (SELECT AVG({column}) FROM {table});"
    )


def _create_table(prompt: str) -> str | None:
    if not re.search(r"\b(?:design|create|write)\b", prompt, re.I) or not re.search(
        r"\btable\b", prompt, re.I
    ):
        return None
    table_hits = re.findall(rf"\btable\s+`?({_IDENT})`?", prompt, re.I)
    if len(table_hits) != 1:
        return None
    columns_match = re.search(
        r"\bcolumns?\s*:?[ \t]*(.+?)(?:\.\s*(?:Write|Return|Provide)\b|\.\s*$|$)",
        prompt,
        re.I | re.S,
    )
    if not columns_match:
        return None
    specs = _split_top_level(columns_match.group(1).strip())
    if not 2 <= len(specs) <= 32:
        return None

    columns: list[tuple[str, str, bool, bool, bool]] = []
    for spec in specs:
        parsed = _column_spec(spec)
        if parsed is None:
            return None
        columns.append(parsed)
    names = [item[0].lower() for item in columns]
    if len(set(names)) != len(names):
        return None

    explicit_refs = re.findall(
        rf"\b({_IDENT})\s*\([^)]*?\bforeign\s+key\s+references\s+"
        rf"({_IDENT})\s*\(\s*({_IDENT})\s*\)",
        prompt,
        re.I,
    )
    refs_by_col: dict[str, tuple[str, str]] = {}
    for column, target_table, target_col in explicit_refs:
        key = column.lower()
        if key in refs_by_col or key not in names:
            return None
        refs_by_col[key] = (target_table, target_col)

    foreign_columns = [name for name, _type, _pk, foreign, _unique in columns if foreign]
    for foreign in foreign_columns:
        key = foreign.lower()
        if key not in refs_by_col:
            if key == "customer_id":
                refs_by_col[key] = ("customers", "id")
            else:
                return None
    if set(refs_by_col) != {column.lower() for column in foreign_columns}:
        return None

    table = table_hits[0]
    if not _safe_identifiers(table, *(item[0] for item in columns)):
        return None
    lines: list[str] = []
    for name, type_sql, primary, _foreign, unique in columns:
        suffix = " PRIMARY KEY" if primary else ""
        if unique:
            suffix += " UNIQUE"
        lines.append(f"    {name} {type_sql}{suffix}")
    for name, _type, _primary, foreign, _unique in columns:
        if foreign:
            target_table, target_col = refs_by_col[name.lower()]
            if not _safe_identifiers(target_table, target_col):
                return None
            lines.append(f"    FOREIGN KEY ({name}) REFERENCES {target_table}({target_col})")
    return f"CREATE TABLE {table} (\n" + ",\n".join(lines) + "\n);"


def _column_spec(spec: str) -> tuple[str, str, bool, bool, bool] | None:
    match = re.fullmatch(rf"\s*`?({_IDENT})`?\s*\((.*)\)\s*", spec, re.I)
    if not match:
        return None
    name, descriptor = match.groups()
    desc = re.sub(
        rf"\breferences\s+{_IDENT}\s*\(\s*{_IDENT}\s*\)",
        "",
        descriptor.lower(),
        flags=re.I,
    )
    desc = " ".join(desc.replace("_", " ").split())
    if not desc or re.search(r"[^a-z0-9,()\s]", desc):
        return None
    primary = "primary key" in desc
    foreign = "foreign key" in desc
    unique = bool(re.search(r"\bunique\b", desc))

    decimal = re.search(r"\bdecimal(?:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\))?\b", desc)
    varchar = re.search(r"\bvarchar(?:\s*\(\s*(\d+)\s*\))?\b", desc)
    type_hits = sum(
        bool(hit)
        for hit in (
            re.search(r"\b(?:int|integer)\b", desc),
            re.search(r"\bdate\b", desc),
            decimal,
            varchar,
            re.search(r"\bboolean\b", desc),
            re.search(r"\btext\b", desc),
        )
    )
    if type_hits > 1:
        return None
    if re.search(r"\b(?:int|integer)\b", desc):
        type_sql = "INT"
    elif re.search(r"\bdate\b", desc):
        type_sql = "DATE"
    elif decimal:
        precision = decimal.group(1) or "10"
        scale = decimal.group(2) or "2"
        type_sql = f"DECIMAL({int(precision)}, {int(scale)})"
    elif varchar:
        type_sql = f"VARCHAR({int(varchar.group(1) or '255')})"
    elif re.search(r"\bboolean\b", desc):
        type_sql = "BOOLEAN"
    elif re.search(r"\btext\b", desc):
        type_sql = "TEXT"
    elif name.lower().endswith("_id") and (primary or foreign):
        type_sql = "INT"
    else:
        return None
    return name, type_sql, primary, foreign, unique


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return []
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    if depth != 0:
        return []
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _safe_identifiers(*values: str) -> bool:
    return all(re.fullmatch(_IDENT, value) for value in values)


def _self_check() -> None:
    assert solve(
        "Write a SQL query to select all columns from a table named `customers` where the "
        "`country` column equals 'Canada'."
    ) == "SELECT * FROM customers WHERE country = 'Canada';"
    assert "SELECT AVG(salary) FROM employees" in (solve(
        "This SQL query should return employees above the average. Fix it.\n"
        "```sql\nSELECT name FROM employees WHERE salary > 50000;\n```"
    ) or "")
    ddl = solve(
        "Design a SQL table `products` with columns product_id (primary key int), "
        "name (varchar), price (decimal)."
    )
    assert ddl and "product_id INT PRIMARY KEY" in ddl and "price DECIMAL(10, 2)" in ddl
    assert solve("Write SQL to delete all rows from users.") is None
    assert solve("Create table pets with columns name (varchar), owner_id (foreign key).") is None
    assert solve("Create table pets with columns name (varchar), mystery (fast).") is None


if __name__ == "__main__":
    _self_check()
    print("SQL solver self-check passed")
