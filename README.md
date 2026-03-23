# cocoindex-code-dafny

Dafny AST-aware chunker for [cocoindex-code](https://github.com/cocoindex-io/cocoindex-code) (`ccc`).

Uses [tree-sitter-dafny](https://github.com/hath995/tree-sitter-dafny) to parse Dafny source files and extract individual declarations (methods, lemmas, functions, predicates, datatypes, classes, traits) as discrete chunks with structured metadata headers — optimized for semantic code search by LLM agents.

## What it does

Instead of generic text chunking, each chunk is a complete Dafny declaration with:

- **Module path** (`Std.Collections.Seq`) — for import resolution
- **Declaration kind** (`lemma`, `ghost predicate`, `function`, etc.)
- **Name** — for direct lookup
- **Contracts** (`requires`, `ensures`) — in the metadata header
- **Doc comments** — preserved for natural language search
- **Full source** — the complete declaration body

Example chunk:

```
// Module: Std.Collections.Seq
// Kind: lemma
// Name: SortedUnique
// requires SortedBy(R, xs)
// requires SortedBy(R, ys)
// requires TotalOrdering(R)
// requires multiset(xs) == multiset(ys)
// ensures xs == ys

lemma SortedUnique<T(!new)>(xs: seq<T>, ys: seq<T>, R: (T, T) -> bool)
    requires SortedBy(R, xs)
    requires SortedBy(R, ys)
    requires TotalOrdering(R)
    requires multiset(xs) == multiset(ys)
    ensures xs == ys
  { ... }
```

## Installation

```bash
pip install cocoindex-code-dafny
```

Or with `uv`:

```bash
uv tool install cocoindex-code --prerelease explicit \
  --with cocoindex-code-dafny
```

## Setup

1. Initialize a ccc project in your Dafny codebase:

```bash
cd /path/to/your/dafny/project
ccc init
```

2. Edit `.cocoindex_code/settings.yml` to include `.dfy` files and register the chunker:

```yaml
include_patterns:
  - '**/*.dfy'

chunkers:
  - ext: dfy
    module: cocoindex_code_dafny:dafny_chunker
```

3. Index and search:

```bash
ccc index
ccc search "lemma about sorted sequences"
```

## Embedding Model

By default, ccc uses a local [sentence-transformers](https://www.sbert.net/) model (`all-MiniLM-L6-v2`) which requires no API key. For better results with Dafny code, we recommend using a code-specific embedding model.

In our testing, [nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code) produced noticeably better search results for Dafny declarations than [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) — particularly for queries involving specification patterns like `requires`/`ensures` clauses and multiset reasoning.

To configure a custom model, edit `~/.cocoindex_code/global_settings.yml`:

```yaml
# Local model via sentence-transformers (recommended for simplicity)
embedding:
  provider: sentence-transformers
  model: nomic-ai/nomic-embed-code
  device: cpu  # or cuda
```

```yaml
# Or via LM Studio / any OpenAI-compatible API
embedding:
  provider: litellm
  model: openai/text-embedding-nomic-embed-code
envs:
  OPENAI_API_BASE: http://localhost:1234/v1
  OPENAI_API_KEY: not-needed
```

See the [cocoindex-code documentation](https://github.com/cocoindex-io/cocoindex-code) for more embedding configuration options including Ollama, OpenAI, and other providers.

**Note:** Changing the embedding model requires re-indexing (`ccc reset && ccc index`) since embeddings from different models are not compatible.

## MCP Server

The chunker works transparently with ccc's MCP server:

```bash
# For Claude Code
claude mcp add cocoindex-code -- ccc mcp

# For Codex
codex mcp add cocoindex-code -- ccc mcp
```

Agents can then search for Dafny declarations using natural language or Dafny code patterns:

```
search("predicate checking if map is injective")
search("multiset(xs) == multiset(f(xs)) && sorted(f(xs))")
search("function to convert Option to Result")
```

## License

MIT
