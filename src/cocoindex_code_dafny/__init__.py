"""Dafny AST-aware chunker for cocoindex-code (ccc).

Extracts individual declarations (methods, lemmas, functions, predicates,
datatypes, classes, traits, etc.) as discrete chunks with metadata headers
for optimal LLM search and retrieval.

Usage in .cocoindex_code/settings.yml::

    chunkers:
      - ext: dfy
        module: cocoindex_code_dafny:dafny_chunker
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from cocoindex_code.chunking import Chunk, TextPosition
except ImportError:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TextPosition:
        byte_offset: int
        char_offset: int
        line: int
        column: int

    @dataclass(frozen=True)
    class Chunk:
        text: str
        start: TextPosition
        end: TextPosition

import tree_sitter_dafny as _tsd
from tree_sitter import Language, Node, Parser

DAFNY_LANGUAGE = Language(_tsd.language())

MAX_CHUNK_SIZE = 1500

DECL_TYPES = {
    "method_decl",
    "function_decl",
    "iterator_decl",
    "constructor_decl",
}

CONTAINER_TYPES = {
    "class_decl",
    "datatype_decl",
    "trait_decl",
    "newtype_decl",
}

SPEC_CLAUSE_TYPES = {
    "requires_clause",
    "ensures_clause",
    "decreases_clause",
    "reads_clause",
    "modifies_clause",
    "invariant_clause",
}


import re

_ATTRIBUTE_RE = re.compile(r'\s*\{:[\w\s]+\}')


def _strip_attributes(text: str) -> str:
    """Remove Dafny attribute annotations like {:verify}, {:induction false}, etc."""
    return _ATTRIBUTE_RE.sub('', text)


def _make_parser() -> Parser:
    return Parser(DAFNY_LANGUAGE)


def _get_module_path(node: Node) -> str:
    parts = []
    current = node.parent
    while current is not None:
        if current.type == "module_definition":
            name_node = current.child_by_field_name("name")
            if name_node:
                parts.append(name_node.text.decode())
        current = current.parent
    parts.reverse()
    return ".".join(parts)


def _get_container_name(node: Node) -> Optional[str]:
    name_node = node.child_by_field_name("name")
    return name_node.text.decode() if name_node else None


def _determine_kind(node: Node) -> str:
    """Determine the declaration kind from AST keywords.

    method_decl can be: method, lemma, twostate lemma, least lemma, greatest lemma
    function_decl can be: function, predicate, ghost function, ghost predicate,
                          twostate function, twostate predicate,
                          least predicate, greatest predicate
    """
    keywords = []
    # Collect all preceding decl_modifiers (e.g., opaque ghost)
    modifiers = []
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "decl_modifier":
        modifiers.append(prev.text.decode().strip())
        prev = prev.prev_named_sibling
    modifiers.reverse()
    keywords.extend(modifiers)

    for child in node.children:
        if not child.is_named and child.type in (
            "method", "lemma", "function", "predicate",
            "ghost", "twostate", "least", "greatest",
            "static", "iterator", "constructor",
        ):
            keywords.append(child.type)
        elif child.is_named:
            break

    return " ".join(keywords) if keywords else node.type


def _get_name(node: Node) -> str:
    name_node = node.child_by_field_name("name")
    return name_node.text.decode() if name_node else "<anonymous>"


def _extract_spec_summary(node: Node) -> list[str]:
    specs = []
    for child in node.children:
        if child.type in SPEC_CLAUSE_TYPES:
            text = child.text.decode().strip()
            specs.append(text)
    return specs


def _get_doc_comment(node: Node) -> Optional[str]:
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "decl_modifier":
        prev = prev.prev_named_sibling

    if prev is not None and prev.type == "block_comment":
        text = prev.text.decode().strip()
        if text.startswith("/**"):
            return text
    if prev is not None and prev.type == "line_comment":
        text = prev.text.decode().strip()
        return text
    return None


def _get_signature_end(node: Node) -> int:
    for child in node.children:
        if child.type in ("block_statement", "function_body"):
            return child.start_byte
    return node.end_byte


def _build_chunk_text(
    module_path: str,
    kind: str,
    name: str,
    doc_comment: Optional[str],
    source: str,
) -> str:
    lines = []
    if module_path:
        lines.append(f"// Module: {module_path}")
    lines.append(f"// Kind: {kind}")
    lines.append(f"// Name: {name}")
    lines.append("")
    if doc_comment:
        lines.append(doc_comment)
    lines.append(source)
    return "\n".join(lines)


def _pos_from_node_start(node: Node, source_bytes: bytes) -> TextPosition:
    row, col = node.start_point
    return TextPosition(
        line=row + 1,
        column=col + 1,
        byte_offset=node.start_byte,
        char_offset=len(source_bytes[:node.start_byte].decode("utf-8", errors="replace")),
    )


def _pos_from_node_end(node: Node, source_bytes: bytes) -> TextPosition:
    row, col = node.end_point
    return TextPosition(
        line=row + 1,
        column=col + 1,
        byte_offset=node.end_byte,
        char_offset=len(source_bytes[:node.end_byte].decode("utf-8", errors="replace")),
    )


def _find_first_modifier(node: Node) -> Node:
    """Walk back through consecutive decl_modifier siblings to find the first one."""
    first = node
    prev = node.prev_named_sibling
    while prev is not None and prev.type == "decl_modifier":
        first = prev
        prev = prev.prev_named_sibling
    return first


def _get_source_with_modifier(node: Node, source_bytes: bytes) -> str:
    """Get source text including any preceding decl_modifiers (opaque, ghost, static, etc.)."""
    first = _find_first_modifier(node)
    if first is not node:
        return source_bytes[first.start_byte:node.end_byte].decode("utf-8", errors="replace")
    return node.text.decode()


def _extract_decl_chunks(
    node: Node,
    source_bytes: bytes,
    module_path: str,
    container_name: Optional[str] = None,
) -> list[Chunk]:
    kind = _determine_kind(node)
    name = _get_name(node)
    doc = _get_doc_comment(node)
    source = _strip_attributes(_get_source_with_modifier(node, source_bytes))

    full_path = module_path
    if container_name:
        full_path = f"{module_path}.{container_name}" if module_path else container_name

    chunk_text = _build_chunk_text(full_path, kind, name, doc, source)

    chunks = []
    start = _pos_from_node_start(node, source_bytes)
    end = _pos_from_node_end(node, source_bytes)

    if len(chunk_text) <= MAX_CHUNK_SIZE:
        chunks.append(Chunk(text=chunk_text, start=start, end=end))
    else:
        sig_end_byte = _get_signature_end(node)
        first_mod = _find_first_modifier(node)
        sig_start = first_mod.start_byte if first_mod is not node else node.start_byte
        sig_source = source_bytes[sig_start:sig_end_byte].decode("utf-8", errors="replace").rstrip()
        sig_text = _build_chunk_text(full_path, kind, name, doc, sig_source)
        chunks.append(Chunk(text=sig_text, start=start, end=end))
        chunks.append(Chunk(text=chunk_text, start=start, end=end))

    return chunks


def _extract_container_chunks(
    node: Node,
    source_bytes: bytes,
    module_path: str,
) -> list[Chunk]:
    kind = node.type.replace("_decl", "").replace("_", " ")
    name = _get_container_name(node) or "<anonymous>"
    doc = _get_doc_comment(node)

    source = node.text.decode()

    # Build a rich header for classes/traits: fields, constructor (full), method signatures
    field_lines: list[str] = []
    constructor_source: Optional[str] = None
    method_sigs: list[str] = []

    members = []
    for child in node.children:
        if child.type == "class_member":
            members.extend(child.children)

    i = 0
    while i < len(members):
        m = members[i]
        if m.type in ("field_decl", "constant_field_decl"):
            first_mod = _find_first_modifier(m)
            if first_mod is not m:
                field_lines.append(source_bytes[first_mod.start_byte:m.end_byte].decode("utf-8", errors="replace").strip())
            else:
                field_lines.append(m.text.decode().strip())
        elif m.type == "constructor_decl":
            constructor_source = _strip_attributes(_get_source_with_modifier(m, source_bytes))
        elif m.type in DECL_TYPES:
            # Build signature: modifiers + keyword + name + params + return + specs
            first_mod = _find_first_modifier(m)
            sig_start = first_mod.start_byte if first_mod is not m else m.start_byte
            sig_end = _get_signature_end(m)
            sig = source_bytes[sig_start:sig_end].decode("utf-8", errors="replace").rstrip()
            # Strip attribute annotations like {:verify}, {:vcs_split_on_every_assert}, etc.
            sig = _strip_attributes(sig)
            method_sigs.append(sig)
        i += 1

    body_start = None
    for child in node.children:
        if child.type == "{":
            body_start = child.start_byte - node.start_byte
            break

    sig = source[:body_start].rstrip() if body_start is not None else source.split("{")[0].rstrip()

    if field_lines or constructor_source or method_sigs:
        parts = [sig + " {"]
        for fl in field_lines:
            parts.append("    " + fl)
        if field_lines and (constructor_source or method_sigs):
            parts.append("")
        if constructor_source:
            for line in constructor_source.splitlines():
                parts.append("    " + line)
            if method_sigs:
                parts.append("")
        for ms in method_sigs:
            for j, line in enumerate(ms.splitlines()):
                parts.append(("    " if j == 0 else "        ") + line)
        parts.append("  }")
        header_source = "\n".join(parts)
    elif body_start is not None:
        header_source = sig + " { ... }"
    else:
        header_source = source.split("{")[0].rstrip() if "{" in source else source

    header_text = _build_chunk_text(module_path, kind, name, doc, header_source)

    chunks = []
    start = _pos_from_node_start(node, source_bytes)
    end = _pos_from_node_end(node, source_bytes)

    # Only emit header chunk if it contains useful info beyond just "class Foo { ... }"
    # Datatypes always have constructors in the header; classes/traits emit if they have
    # extends, fields, or doc comments
    def _has_fields(n: Node) -> bool:
        for c in n.children:
            if c.type in ("field_decl", "constant_field_decl"):
                return True
            if c.type == "class_member":
                if any(m.type in ("field_decl", "constant_field_decl") for m in c.children):
                    return True
        return False

    has_useful_header = (
        node.type == "datatype_decl"
        or node.type == "newtype_decl"
        or doc is not None
        or "extends" in header_source
        or _has_fields(node)
    )
    if has_useful_header:
        chunks.append(Chunk(text=header_text, start=start, end=end))

    for child in node.children:
        if child.type == "class_member":
            for member in child.children:
                if member.type in DECL_TYPES:
                    chunks.extend(
                        _extract_decl_chunks(member, source_bytes, module_path, name)
                    )
        elif child.type in DECL_TYPES:
            chunks.extend(
                _extract_decl_chunks(child, source_bytes, module_path, name)
            )

    return chunks


def _build_module_header(
    node: Node,
    source_bytes: bytes,
    module_path: str,
    module_node: Node,
) -> Optional[Chunk]:
    """Build a header chunk for a module showing its API surface.

    Useful for abstract/refinement modules that act like interfaces.
    Only emits a header if the module has declarations worth summarizing.
    """
    children = [c for c in node.children if c.is_named]
    if not children:
        return None

    type_lines: list[str] = []
    const_lines: list[str] = []
    method_sigs: list[str] = []
    has_import_or_export = False

    i = 0
    while i < len(children):
        c = children[i]
        if c.type in ("module_import", "module_export"):
            has_import_or_export = True
        elif c.type == "synonym_type_decl":
            type_lines.append(c.text.decode().strip())
        elif c.type in ("constant_field_decl", "field_decl"):
            first_mod = _find_first_modifier(c)
            if first_mod is not c:
                const_lines.append(source_bytes[first_mod.start_byte:c.end_byte].decode("utf-8", errors="replace").strip())
            else:
                const_lines.append(c.text.decode().strip())
        elif c.type in DECL_TYPES:
            first_mod = _find_first_modifier(c)
            sig_start = first_mod.start_byte if first_mod is not c else c.start_byte
            sig_end = _get_signature_end(c)
            sig = source_bytes[sig_start:sig_end].decode("utf-8", errors="replace").rstrip()
            sig = _strip_attributes(sig)
            method_sigs.append(sig)
        elif c.type == "decl_modifier" and i + 1 < len(children):
            # Skip, will be picked up by _find_first_modifier on next node
            pass
        elif c.type in CONTAINER_TYPES:
            name = _get_container_name(c) or "<anonymous>"
            kind = c.type.replace("_decl", "").replace("_", " ")
            type_lines.append(f"{kind} {name}")
        i += 1

    # Only emit header if the module has meaningful API surface
    if not type_lines and not const_lines and not method_sigs:
        return None

    # Build module declaration line with abstract modifier and refines clause
    mod_prefix = ""
    first_mod = _find_first_modifier(module_node)
    if first_mod is not module_node:
        mod_prefix = source_bytes[first_mod.start_byte:module_node.start_byte].decode("utf-8", errors="replace").strip() + " "

    refines = ""
    for c in module_node.children:
        if c.type == "refines_clause":
            refines = " " + c.text.decode().strip()
            break

    parts = [f"{mod_prefix}module {module_path}{refines} {{"]
    for tl in type_lines:
        parts.append("    " + tl)
    if type_lines and (const_lines or method_sigs):
        parts.append("")
    for cl in const_lines:
        parts.append("    " + cl)
    if const_lines and method_sigs:
        parts.append("")
    for ms in method_sigs:
        for j, line in enumerate(ms.splitlines()):
            parts.append(("    " if j == 0 else "        ") + line)
    parts.append("}")
    header_source = "\n".join(parts)

    # Get doc comment from before the module_definition
    doc = _get_doc_comment(module_node)

    kind = f"{mod_prefix.strip()} module".strip() if mod_prefix.strip() else "module"
    header_text = _build_chunk_text(module_path, kind, module_path.split(".")[-1] if module_path else "", doc, header_source)

    start = _pos_from_node_start(module_node, source_bytes)
    end = _pos_from_node_end(module_node, source_bytes)
    return Chunk(text=header_text, start=start, end=end)


def _walk_module_body(
    node: Node,
    source_bytes: bytes,
    module_path: str,
    module_node: Optional[Node] = None,
) -> list[Chunk]:
    chunks = []

    # Emit a module header chunk if the module has API surface
    if module_node is not None:
        header = _build_module_header(node, source_bytes, module_path, module_node)
        if header is not None:
            chunks.append(header)

    i = 0
    children = list(node.children)

    while i < len(children):
        child = children[i]

        if child.type == "module_definition":
            name_node = child.child_by_field_name("name")
            child_module = name_node.text.decode() if name_node else ""
            for sub in child.children:
                if sub.type == "module_body":
                    chunks.extend(_walk_module_body(sub, source_bytes, child_module, child))
            i += 1
            continue

        if child.type in CONTAINER_TYPES:
            chunks.extend(_extract_container_chunks(child, source_bytes, module_path))
            i += 1
            continue

        if child.type in DECL_TYPES:
            chunks.extend(_extract_decl_chunks(child, source_bytes, module_path))
            i += 1
            continue

        if child.type == "decl_modifier":
            if i + 1 < len(children) and children[i + 1].type in DECL_TYPES:
                chunks.extend(
                    _extract_decl_chunks(children[i + 1], source_bytes, module_path)
                )
                i += 2
                continue
            elif i + 1 < len(children) and children[i + 1].type in CONTAINER_TYPES:
                chunks.extend(
                    _extract_container_chunks(children[i + 1], source_bytes, module_path)
                )
                i += 2
                continue

        if child.type in ("constant_field_decl", "synonym_type_decl", "field_decl"):
            kind = child.type.replace("_decl", "").replace("_", " ")
            name = _get_name(child)
            source = child.text.decode()
            text = _build_chunk_text(module_path, kind, name, None, source)
            start = _pos_from_node_start(child, source_bytes)
            end = _pos_from_node_end(child, source_bytes)
            chunks.append(Chunk(text=text, start=start, end=end))
            i += 1
            continue

        i += 1

    return chunks


def _build_broken_module_header(
    children: list[Node],
    name_idx: int,
    source_bytes: bytes,
    module_path: str,
) -> Optional[Chunk]:
    """Build a module header from flattened root-level children (broken parse)."""
    type_lines: list[str] = []
    const_lines: list[str] = []
    method_sigs: list[str] = []

    # Detect abstract modifier
    mod_prefix = ""
    j = name_idx - 1
    while j >= 0 and children[j].type in ("module", "decl_modifier"):
        if children[j].type == "decl_modifier":
            mod_prefix = children[j].text.decode().strip() + " " + mod_prefix
        j -= 1

    # Scan remaining children for declarations
    i = name_idx + 1
    while i < len(children):
        c = children[i]
        if c.type == "synonym_type_decl":
            type_lines.append(c.text.decode().strip())
        elif c.type in ("constant_field_decl", "field_decl"):
            first_mod = _find_first_modifier(c)
            if first_mod is not c:
                const_lines.append(source_bytes[first_mod.start_byte:c.end_byte].decode("utf-8", errors="replace").strip())
            else:
                const_lines.append(c.text.decode().strip())
        elif c.type in DECL_TYPES:
            first_mod = _find_first_modifier(c)
            sig_start = first_mod.start_byte if first_mod is not c else c.start_byte
            sig_end = _get_signature_end(c)
            sig = source_bytes[sig_start:sig_end].decode("utf-8", errors="replace").rstrip()
            sig = _strip_attributes(sig)
            method_sigs.append(sig)
        elif c.type == "decl_modifier" and i + 1 < len(children) and children[i + 1].type in DECL_TYPES:
            pass  # will be picked up via _find_first_modifier
        i += 1

    if not type_lines and not const_lines and not method_sigs:
        return None

    parts = [f"{mod_prefix}module {module_path} {{"]
    for tl in type_lines:
        parts.append("    " + tl)
    if type_lines and (const_lines or method_sigs):
        parts.append("")
    for cl in const_lines:
        parts.append("    " + cl)
    if const_lines and method_sigs:
        parts.append("")
    for ms in method_sigs:
        for k, line in enumerate(ms.splitlines()):
            parts.append(("    " if k == 0 else "        ") + line)
    parts.append("}")
    header_source = "\n".join(parts)

    kind = f"{mod_prefix.strip()} module".strip() if mod_prefix.strip() else "module"
    header_text = _build_chunk_text(module_path, kind, module_path.split(".")[-1], None, header_source)

    start = TextPosition(byte_offset=0, char_offset=0, line=1, column=1)
    end = TextPosition(
        byte_offset=len(source_bytes),
        char_offset=len(source_bytes.decode("utf-8", errors="replace")),
        line=source_bytes.count(b"\n") + 1,
        column=1,
    )
    return Chunk(text=header_text, start=start, end=end)


def chunk_file(content: str) -> list[Chunk]:
    """Parse a Dafny file and return AST-aware chunks."""
    parser = _make_parser()
    source_bytes = content.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks = []
    children = list(root.children)

    # Detect broken module structure: when parse errors cause module_definition
    # to be flattened into root-level tokens (module + qualified_name + { + decls)
    # Reconstruct the module context for proper chunking.
    broken_module_path = ""
    i = 0
    while i < len(children):
        child = children[i]

        if child.type == "module_definition":
            name_node = child.child_by_field_name("name")
            module_path = name_node.text.decode() if name_node else ""
            for sub in child.children:
                if sub.type == "module_body":
                    chunks.extend(_walk_module_body(sub, source_bytes, module_path, child))
            i += 1
        elif child.type == "qualified_name" and i > 0 and children[i - 1].type in ("module", "decl_modifier"):
            # Broken module_definition — extract module name and use as context
            broken_module_path = child.text.decode()
            # Build a module header from the remaining root-level declarations
            header = _build_broken_module_header(children, i, source_bytes, broken_module_path)
            if header is not None:
                chunks.append(header)
            i += 1
        elif child.type in CONTAINER_TYPES:
            chunks.extend(_extract_container_chunks(child, source_bytes, broken_module_path))
            i += 1
        elif child.type in DECL_TYPES:
            chunks.extend(_extract_decl_chunks(child, source_bytes, broken_module_path))
            i += 1
        elif child.type == "decl_modifier":
            if i + 1 < len(children) and children[i + 1].type in DECL_TYPES:
                chunks.extend(
                    _extract_decl_chunks(children[i + 1], source_bytes, broken_module_path)
                )
                i += 2
            elif i + 1 < len(children) and children[i + 1].type in CONTAINER_TYPES:
                chunks.extend(
                    _extract_container_chunks(children[i + 1], source_bytes, broken_module_path)
                )
                i += 2
            else:
                i += 1
        elif child.type in ("constant_field_decl", "synonym_type_decl", "field_decl"):
            kind = child.type.replace("_decl", "").replace("_", " ")
            name = _get_name(child)
            source = child.text.decode()
            text = _build_chunk_text(broken_module_path, kind, name, None, source)
            start = _pos_from_node_start(child, source_bytes)
            end = _pos_from_node_end(child, source_bytes)
            chunks.append(Chunk(text=text, start=start, end=end))
            i += 1
        elif child.type == "module_body":
            chunks.extend(_walk_module_body(child, source_bytes, broken_module_path))
            i += 1
        else:
            i += 1

    if not chunks and content.strip():
        chunks.append(Chunk(
            text=content,
            start=TextPosition(byte_offset=0, char_offset=0, line=1, column=1),
            end=TextPosition(
                byte_offset=len(source_bytes),
                char_offset=len(content),
                line=content.count("\n") + 1,
                column=len(content.rsplit("\n", 1)[-1]) + 1,
            ),
        ))

    return chunks


def dafny_chunker(path: Path, content: str) -> tuple[str | None, list[Chunk]]:
    """ccc-compatible chunker entry point.

    Returns:
        ("dafny", list_of_chunks) — language override + chunks
    """
    return ("dafny", chunk_file(content))
