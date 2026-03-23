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
    prev = node.prev_named_sibling
    if prev is not None and prev.type == "decl_modifier":
        keywords.append(prev.text.decode().strip())

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
    specs: list[str],
) -> str:
    lines = []
    if module_path:
        lines.append(f"// Module: {module_path}")
    lines.append(f"// Kind: {kind}")
    lines.append(f"// Name: {name}")
    for spec in specs:
        lines.append(f"// {spec}")
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


def _extract_decl_chunks(
    node: Node,
    source_bytes: bytes,
    module_path: str,
    container_name: Optional[str] = None,
) -> list[Chunk]:
    kind = _determine_kind(node)
    name = _get_name(node)
    doc = _get_doc_comment(node)
    specs = _extract_spec_summary(node)
    source = node.text.decode()

    full_path = module_path
    if container_name:
        full_path = f"{module_path}.{container_name}" if module_path else container_name

    chunk_text = _build_chunk_text(full_path, kind, name, doc, source, specs)

    chunks = []
    start = _pos_from_node_start(node, source_bytes)
    end = _pos_from_node_end(node, source_bytes)

    if len(chunk_text) <= MAX_CHUNK_SIZE:
        chunks.append(Chunk(text=chunk_text, start=start, end=end))
    else:
        sig_end_byte = _get_signature_end(node)
        sig_source = source_bytes[node.start_byte:sig_end_byte].decode("utf-8", errors="replace").rstrip()
        sig_text = _build_chunk_text(full_path, kind, name, doc, sig_source, specs)
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
    body_start = None
    for child in node.children:
        if child.type == "{":
            body_start = child.start_byte - node.start_byte
            break

    if body_start is not None:
        header_source = source[:body_start].rstrip() + " { ... }"
    else:
        header_source = source.split("{")[0].rstrip() if "{" in source else source

    header_text = _build_chunk_text(module_path, kind, name, doc, header_source, [])

    chunks = []
    start = _pos_from_node_start(node, source_bytes)
    end = _pos_from_node_end(node, source_bytes)
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


def _walk_module_body(
    node: Node,
    source_bytes: bytes,
    module_path: str,
) -> list[Chunk]:
    chunks = []
    i = 0
    children = list(node.children)

    while i < len(children):
        child = children[i]

        if child.type == "module_definition":
            name_node = child.child_by_field_name("name")
            child_module = name_node.text.decode() if name_node else ""
            for sub in child.children:
                if sub.type == "module_body":
                    chunks.extend(_walk_module_body(sub, source_bytes, child_module))
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
            text = _build_chunk_text(module_path, kind, name, None, source, [])
            start = _pos_from_node_start(child, source_bytes)
            end = _pos_from_node_end(child, source_bytes)
            chunks.append(Chunk(text=text, start=start, end=end))
            i += 1
            continue

        i += 1

    return chunks


def chunk_file(content: str) -> list[Chunk]:
    """Parse a Dafny file and return AST-aware chunks."""
    parser = _make_parser()
    source_bytes = content.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks = []

    for child in root.children:
        if child.type == "module_definition":
            name_node = child.child_by_field_name("name")
            module_path = name_node.text.decode() if name_node else ""
            for sub in child.children:
                if sub.type == "module_body":
                    chunks.extend(_walk_module_body(sub, source_bytes, module_path))
        elif child.type in CONTAINER_TYPES:
            chunks.extend(_extract_container_chunks(child, source_bytes, ""))
        elif child.type in DECL_TYPES:
            chunks.extend(_extract_decl_chunks(child, source_bytes, ""))
        elif child.type == "module_body":
            chunks.extend(_walk_module_body(child, source_bytes, ""))

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
