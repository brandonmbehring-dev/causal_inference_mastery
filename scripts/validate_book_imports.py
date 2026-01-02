#!/usr/bin/env python3
"""
Validate Book Imports Against Module Exports.

This script parses LaTeX book chapters for minted Python code blocks,
extracts import statements, and verifies they match actual module exports.

Usage:
    python scripts/validate_book_imports.py
    python scripts/validate_book_imports.py --chapter ch26_timeseries
    python scripts/validate_book_imports.py --verbose

Output:
    - Coverage report: which modules are referenced
    - Missing exports: imports that don't exist
    - Unused exports: module exports not in book
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    names: list[str]
    chapter: str
    line_number: int


@dataclass
class ValidationResult:
    """Result of validating book imports."""

    total_imports: int = 0
    valid_imports: int = 0
    invalid_imports: list[tuple[ImportInfo, str]] = field(default_factory=list)
    modules_referenced: set[str] = field(default_factory=set)
    exports_used: dict[str, set[str]] = field(default_factory=dict)
    exports_unused: dict[str, set[str]] = field(default_factory=dict)


def find_minted_blocks(tex_content: str) -> list[tuple[int, str]]:
    """Extract Python minted blocks from LaTeX content.

    Args:
        tex_content: LaTeX source content.

    Returns:
        List of (line_number, code_block) tuples.
    """
    pattern = r"\\begin\{minted\}(?:\[.*?\])?\{python\}(.*?)\\end\{minted\}"
    blocks = []

    # Track line numbers
    lines = tex_content.split("\n")
    current_pos = 0

    for match in re.finditer(pattern, tex_content, re.DOTALL):
        # Find line number
        start_pos = match.start()
        line_num = tex_content[:start_pos].count("\n") + 1
        code = match.group(1).strip()
        blocks.append((line_num, code))

    return blocks


def extract_imports(code: str) -> list[tuple[str, list[str]]]:
    """Extract import statements from Python code.

    Args:
        code: Python source code.

    Returns:
        List of (module, [imported_names]) tuples.
    """
    imports = []

    # Handle incomplete code gracefully
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fall back to regex for incomplete snippets
        return extract_imports_regex(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, [alias.name]))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = [alias.name for alias in node.names]
                imports.append((node.module, names))

    return imports


def extract_imports_regex(code: str) -> list[tuple[str, list[str]]]:
    """Fallback regex-based import extraction.

    Args:
        code: Python source code (may be incomplete).

    Returns:
        List of (module, [imported_names]) tuples.
    """
    imports = []

    # Match: from module import name1, name2, ...
    from_pattern = r"from\s+([\w.]+)\s+import\s+\(?([\w\s,]+)\)?"
    for match in re.finditer(from_pattern, code):
        module = match.group(1)
        names_str = match.group(2)
        names = [n.strip() for n in names_str.split(",") if n.strip()]
        imports.append((module, names))

    # Match: import module
    import_pattern = r"^import\s+([\w.]+)"
    for match in re.finditer(import_pattern, code, re.MULTILINE):
        module = match.group(1)
        imports.append((module, [module.split(".")[-1]]))

    return imports


def get_module_exports(module_name: str) -> set[str] | None:
    """Get exports (__all__) from a module.

    Args:
        module_name: Fully qualified module name.

    Returns:
        Set of exported names, or None if module doesn't exist.
    """
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "__all__"):
            return set(module.__all__)
        else:
            # Return public names if no __all__
            return {name for name in dir(module) if not name.startswith("_")}
    except (ImportError, ModuleNotFoundError):
        return None


def validate_import(module: str, names: list[str]) -> tuple[bool, str]:
    """Validate that imported names exist in module.

    Args:
        module: Module name.
        names: List of names to import.

    Returns:
        (is_valid, error_message) tuple.
    """
    # Skip non-causal_inference imports
    if not module.startswith("causal_inference"):
        return True, ""

    exports = get_module_exports(module)
    if exports is None:
        return False, f"Module '{module}' not found"

    missing = [n for n in names if n not in exports]
    if missing:
        return False, f"Names not exported: {missing}"

    return True, ""


def validate_chapter(
    tex_path: Path, verbose: bool = False
) -> tuple[list[ImportInfo], list[tuple[ImportInfo, str]]]:
    """Validate all imports in a chapter.

    Args:
        tex_path: Path to .tex file.
        verbose: Print progress.

    Returns:
        (valid_imports, invalid_imports) tuple.
    """
    content = tex_path.read_text()
    chapter_name = tex_path.stem

    valid = []
    invalid = []

    for line_num, code in find_minted_blocks(content):
        imports = extract_imports(code)

        for module, names in imports:
            info = ImportInfo(
                module=module, names=names, chapter=chapter_name, line_number=line_num
            )

            is_valid, error = validate_import(module, names)
            if is_valid:
                valid.append(info)
                if verbose:
                    print(f"  ✓ {module}: {names}")
            else:
                invalid.append((info, error))
                if verbose:
                    print(f"  ✗ {module}: {error}")

    return valid, invalid


def collect_module_coverage(
    valid_imports: list[ImportInfo],
) -> dict[str, set[str]]:
    """Collect which exports are used from each module.

    Args:
        valid_imports: List of valid import infos.

    Returns:
        Dict mapping module -> set of used names.
    """
    coverage: dict[str, set[str]] = {}

    for info in valid_imports:
        if not info.module.startswith("causal_inference"):
            continue

        if info.module not in coverage:
            coverage[info.module] = set()
        coverage[info.module].update(info.names)

    return coverage


def compute_unused_exports(
    used: dict[str, set[str]]
) -> dict[str, set[str]]:
    """Find exports not used in the book.

    Args:
        used: Dict of module -> used names.

    Returns:
        Dict of module -> unused names.
    """
    unused = {}

    for module, used_names in used.items():
        exports = get_module_exports(module)
        if exports:
            unused_names = exports - used_names
            if unused_names:
                unused[module] = unused_names

    return unused


def main() -> int:
    """Main entry point."""
    # Ensure paths are set up for module imports
    # Note: Some source files incorrectly use 'from src.causal_inference...'
    # so we need both the project root AND src/ in the path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_dir = project_root / "src"

    # Add project root first (for 'from src.causal_inference...' imports)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    # Add src dir second (for proper 'from causal_inference...' imports)
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    # Force reimport of any cached module lookups
    importlib.invalidate_caches()

    parser = argparse.ArgumentParser(description="Validate book imports.")
    parser.add_argument("--chapter", help="Validate specific chapter only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--book-dir",
        default="book/chapters",
        help="Book chapters directory",
    )
    args = parser.parse_args()

    book_dir = Path(args.book_dir)
    if not book_dir.exists():
        print(f"Error: Book directory '{book_dir}' not found")
        return 1

    # Find chapters
    if args.chapter:
        patterns = [f"**/{args.chapter}.tex"]
    else:
        patterns = ["**/*.tex"]

    tex_files = []
    for pattern in patterns:
        tex_files.extend(book_dir.glob(pattern))

    if not tex_files:
        print("No .tex files found")
        return 1

    print(f"Validating {len(tex_files)} chapters...\n")

    all_valid: list[ImportInfo] = []
    all_invalid: list[tuple[ImportInfo, str]] = []

    for tex_file in sorted(tex_files):
        if args.verbose:
            print(f"\n{tex_file.stem}:")

        valid, invalid = validate_chapter(tex_file, verbose=args.verbose)
        all_valid.extend(valid)
        all_invalid.extend(invalid)

    # Compute coverage
    used = collect_module_coverage(all_valid)
    unused = compute_unused_exports(used)

    # Report
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"\nTotal imports checked: {len(all_valid) + len(all_invalid)}")
    print(f"Valid imports: {len(all_valid)}")
    print(f"Invalid imports: {len(all_invalid)}")

    if all_invalid:
        print("\n--- INVALID IMPORTS ---")
        for info, error in all_invalid:
            print(f"  {info.chapter}:{info.line_number}")
            print(f"    from {info.module} import {', '.join(info.names)}")
            print(f"    Error: {error}")

    print(f"\n--- MODULE COVERAGE ({len(used)} modules) ---")
    for module in sorted(used.keys()):
        exports = get_module_exports(module)
        n_exports = len(exports) if exports else 0
        n_used = len(used[module])
        pct = (n_used / n_exports * 100) if n_exports > 0 else 0
        print(f"  {module}: {n_used}/{n_exports} ({pct:.0f}%)")

    if unused and args.verbose:
        print("\n--- UNUSED EXPORTS ---")
        for module, names in sorted(unused.items()):
            print(f"  {module}:")
            for name in sorted(names)[:10]:  # Limit output
                print(f"    - {name}")
            if len(names) > 10:
                print(f"    ... and {len(names) - 10} more")

    # Summary
    print("\n" + "=" * 60)
    if all_invalid:
        print(f"FAILED: {len(all_invalid)} invalid imports found")
        return 1
    else:
        print("PASSED: All imports valid")
        return 0


if __name__ == "__main__":
    sys.exit(main())
