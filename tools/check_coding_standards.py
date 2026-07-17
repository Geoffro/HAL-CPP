#!/usr/bin/env python3
"""
Heuristic checker for a handful of docs/coding_standards.md rules that have
no clang-format/clang-tidy equivalent:

  - "Avoid anonymous namespaces. It should either be typed as a project
    namespace or within the project namespace." [anonymous-namespace]
  - "Do not use 'default', require an implementation even if it is just {}"
    (i.e. no `= default;` on constructors/destructors/assignment operators).
    [defaulted-function]
  - "For initialization, never use Type myVar{};, use Type myVar = {};
    instead." (direct-list-init on a variable declaration, as opposed to a
    constructor initializer-list entry like `, m_params{0.4f, ...}`, which
    this rule does not apply to). [direct-brace-init]
  - "Avoid defining structs inside classes" / "Do not define structs in the
    body of a class." [nested-struct]
  - "prefer to use a logger with severity levels" (Error Handling), extended
    to its natural conclusion: if there's a central Logger, direct
    printf/cout/cerr anywhere bypasses it. This check runs on every scanned
    file with no built-in exemption -- including Logger.cpp's own
    implementation and tools/'s CLI utilities, which legitimately need raw
    output. Those are handled as explicit, visible entries in
    .check_coding_standards.json (one per call site) rather than a silent
    directory/filename exemption baked into the check itself. [raw-output]
  - "Pointers as members use the m_p prefix, otherwise p" -- a raw pointer
    variable/parameter/member should be `p`/`pp` + PascalCase (`pDevice`,
    `ppExtensions`), plus the usual `m_` member prefix for a pointer member
    (`m_pDevice`). Deliberately NOT enforced by .clang-tidy's
    readability-identifier-naming: that check classifies by canonical
    (desugared) type, so opaque handle typedefs (VkCommandBuffer,
    VmaAllocator, VkBuffer, ...) -- pointers under the hood but used bare,
    with no prefix, throughout this codebase -- are indistinguishable from
    real pointers to it (a full-codebase run flagged 99/99 such
    "violations", all false positives; see .clang-tidy's comments). This
    check works on source text instead: it only looks at declarations with
    a literal `*`, so Vulkan/VMA handles (declared bare) and smart pointers
    (std::unique_ptr<T> m_window, also no `*`) never match, no exclusion
    list needed. [pointer-naming]
  - "For compound boolean expressions, parenthesize each comparison or
    logical unit." Checked against all 36 real compound if/while conditions
    in src/ before writing this: the rule as actually practiced is narrower
    than "wrap every operand" -- bare booleans (`!initialized`, `flag`,
    `Foo.Bar()`) are consistently left unwrapped, only an operand containing
    a comparison operator (==, !=, <, >, <=, >=) needs its own enclosing
    parens. This check finds each if(...)/while(...) condition (joining
    multi-line conditions), splits it into operands at top-level &&/||, and
    flags any operand with a top-level comparison that isn't fully wrapped.
    Single-operand conditions (no &&/||) are skipped -- the rule is about
    *compound* expressions. Two real hazards found via full-codebase
    validation runs, not guesswork: static_cast<T>(...)/reinterpret_cast<T>/
    const_cast<T>/dynamic_cast<T>'s <...> looks like a comparison textually,
    so that span is masked out before scanning (no other template-angle-
    bracket usage appears inside a boolean condition anywhere in this
    codebase, so this is scoped to just those four keywords rather than
    general template syntax); and `->` / `<<` / `>>` each contain a `<` or
    `>` character that isn't a comparison (e.g.
    `!m_vulkanContext->IsDeviceLost()` was a false positive on first run),
    so those three tokens are excluded before matching a bare `<`/`>`.
    [compound-boolean-parens]
  - "When a compound boolean condition is wrapped across multiple lines,
    put each top-level &&/|| operand on its own line, with the operator
    trailing at the end of the previous line." clang-format has no
    equivalent: its line-breaking is column-width driven (pack onto one
    line whenever it fits under ColumnLimit), and BreakBeforeBinaryOperators
    only controls where the operator goes once a break already happens, not
    whether a condition is single-line or one-operand-per-line -- so
    clang-format will happily re-pack a wrapped condition's first two
    operands onto one line as long as the result still fits, exactly the
    regression this check exists to catch. It reuses the same condition/
    operand split as compound-boolean-parens above, but only looks at
    conditions that already span multiple lines (a condition clang-format
    kept on one line is untouched either way -- this rule is about layout
    *within* an already-wrapped condition, not about forcing a wrap).
    For every operator boundary inside such a condition it checks two
    things: the operator itself has no blank/newline before it (i.e. it
    trails the previous operand's line rather than leading its own), and
    the next operand's first non-whitespace character is preceded by a
    newline (i.e. it starts a fresh line rather than sharing the operator's
    line). [compound-boolean-layout]

This is a best-effort linter, not a formatter: it flags likely violations
for a human (or Claude) to fix, it does not rewrite files. Each check is
line/brace-based rather than a real parser -- it distinguishes a
declaration's `Type *name`/`Type name{...}` from an expression like
`count * scale` or `return MyStruct{1, 2}` by requiring the token in the
type position to look like a type (CamelCase, a fixed-width/primitive
keyword, or a leading underscore as used by vendored C headers like
_freenect_device) -- false positives and false negatives are both possible
on unusual code shapes, read the flagged lines before "fixing" them.

Known, intentional exceptions (e.g. `argv` in a `main()` signature not
taking the `pp` prefix, or Logger.cpp's own use of std::cout/std::cerr) are
suppressed via .check_coding_standards.json at the repo root rather than
silently special-cased in code, so every suppression is visible and
requires a reason. Each ignore entry suppresses exactly one violation --
matching is consumed after first use, so one entry never silently blankets
every occurrence of, say, a repeated identifier across a file; a recurring
violation needs one entry per site. See that file for the format;
suppressed hits are still printed (prefixed "[suppressed: <check>]") so
they stay auditable instead of disappearing, and any ignore entry that
never matched anything prints a warning so the config can't quietly rot.

This script has no repo-specific paths baked in -- it takes the paths to
scan as required arguments, and resolves the ignore-config file relative to
a --root that defaults to the current working directory. It can be copied
into another C++ codebase and pointed at that codebase's own directories
and its own .check_coding_standards.json.

Usage:
    python3 check_coding_standards.py check [options] path [path ...]
    python3 check_coding_standards.py verify-config [options]

`check` is the linter above. `verify-config` validates the ignore file on
its own, independent of any particular scan: well-formed JSON, every entry
has the required fields (check/file/reason) and a known check id, its file
exists, and -- by re-running the checks against every file it references --
it still matches a real, current violation rather than one that's since
been fixed, renamed, or moved to a different line.

Run either subcommand with --help for its full option list.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PRIMITIVE_TYPES = {'void', 'bool', 'char', 'int', 'float', 'double', 'short', 'long', 'unsigned', 'signed', 'wchar_t'}

ANON_NAMESPACE_INLINE = re.compile(r'^\s*namespace\s*\{')
ANON_NAMESPACE_BARE   = re.compile(r'^\s*namespace\s*$')
DEFAULTED_FUNCTION    = re.compile(r'=\s*default\s*;')
DIRECT_BRACE_INIT     = re.compile(
    r'^\s*(?:static\s+)?(?:const(?:expr)?\s+)?(?P<type>[A-Za-z_][\w:<>, ]*[\w>])\s+(?P<var>[A-Za-z_]\w*)\s*\{'
)
CLASS_OR_STRUCT_DECL  = re.compile(r'^\s*(?P<kind>class|struct)\s+(?P<name>[A-Za-z_]\w*)\b')
RAW_OUTPUT            = re.compile(r'\b(?:printf|fprintf|puts|std::cout|std::cerr|std::clog)\b')

# `Type *name` / `Type **name`, immediately followed by a declaration-ish
# delimiter (end of parameter, end of statement, or start of initializer).
POINTER_DECL = re.compile(r'\b(?P<type>[A-Za-z_][\w:<>]*)\s+(?P<stars>\*+)\s*(?P<name>[A-Za-z_]\w*)\s*(?=[,)=;])')

IF_WHILE            = re.compile(r'\b(?:if|while)\s*\(')
CAST_KEYWORDS       = ('static_cast', 'reinterpret_cast', 'const_cast', 'dynamic_cast')
COMPARISON_OPS      = ('==', '!=', '<=', '>=', '<', '>')
NON_COMPARISON_OPS  = ('->', '<<', '>>')

# Single source of truth for check ids, so `verify-config` can't drift out of
# sync with the ids the checks below actually emit.
CHECK_ANONYMOUS_NAMESPACE = 'anonymous-namespace'
CHECK_DEFAULTED_FUNCTION  = 'defaulted-function'
CHECK_DIRECT_BRACE_INIT   = 'direct-brace-init'
CHECK_NESTED_STRUCT       = 'nested-struct'
CHECK_RAW_OUTPUT          = 'raw-output'
CHECK_POINTER_NAMING      = 'pointer-naming'
CHECK_COMPOUND_BOOLEAN    = 'compound-boolean-parens'
CHECK_COMPOUND_BOOLEAN_LAYOUT = 'compound-boolean-layout'

KNOWN_CHECKS = {
    CHECK_ANONYMOUS_NAMESPACE,
    CHECK_DEFAULTED_FUNCTION,
    CHECK_DIRECT_BRACE_INIT,
    CHECK_NESTED_STRUCT,
    CHECK_RAW_OUTPUT,
    CHECK_POINTER_NAMING,
    CHECK_COMPOUND_BOOLEAN,
    CHECK_COMPOUND_BOOLEAN_LAYOUT,
}

REQUIRED_ENTRY_FIELDS = ('check', 'file', 'reason')


def strip_line_comment(line):
    idx = line.find('//')
    return line if idx == -1 else line[:idx]


def strip_line_comment_keep_newline(line):
    idx = line.find('//')

    if idx == -1:
        return line

    return line[:idx] + ('\n' if line.endswith('\n') else '')


def looks_like_type(token):
    base = token.rsplit('::', 1)[-1]

    if base in PRIMITIVE_TYPES:
        return True

    if base.endswith('_t'):
        return True

    if base.startswith('_'):
        return True

    return base[:1].isupper()


def expected_pointer_prefix(name, stars):
    stars_prefix = 'p' * len(stars)
    return f'm_{stars_prefix}' if name.startswith('m_') else stars_prefix


def is_valid_pointer_name(name, stars):
    prefix = expected_pointer_prefix(name, stars)

    if not name.startswith(prefix):
        return False

    rest = name[len(prefix):]

    return rest[:1].isupper()


def check_anonymous_namespaces(lines, issues):
    for lineno, raw_line in enumerate(lines, start=1):
        line = strip_line_comment(raw_line)

        if ANON_NAMESPACE_INLINE.match(line) or ANON_NAMESPACE_BARE.match(line):
            issues.append((lineno, CHECK_ANONYMOUS_NAMESPACE, 'anonymous namespace (name it, or fold into the project namespace)'))


def check_defaulted_functions(lines, issues):
    for lineno, raw_line in enumerate(lines, start=1):
        line = strip_line_comment(raw_line)

        if DEFAULTED_FUNCTION.search(line):
            issues.append((lineno, CHECK_DEFAULTED_FUNCTION, "'= default' used (provide an implementation, even if just {})"))


def check_direct_brace_init(lines, issues):
    for lineno, raw_line in enumerate(lines, start=1):
        line = strip_line_comment(raw_line)
        m    = DIRECT_BRACE_INIT.match(line)

        if m and looks_like_type(m.group('type')):
            var = m.group('var')
            issues.append((lineno, CHECK_DIRECT_BRACE_INIT, f"'{var}{{...}}' uses direct-list-init (use '{var} = {{...}};' instead)"))


def check_nested_structs(lines, issues):
    stack   = []
    depth   = 0
    pending = None

    for lineno, raw_line in enumerate(lines, start=1):
        line = strip_line_comment(raw_line)
        m    = CLASS_OR_STRUCT_DECL.match(line)

        if m and pending is None:
            rest = line[m.end():]

            if not (('{' not in rest) and (';' in rest)):
                pending = (m.group('kind'), m.group('name'), lineno)

        for ch in line:
            if ch == '{':
                depth += 1

                if pending is not None:
                    kind, name, decl_lineno = pending

                    if (kind == 'struct') and any(k == 'class' for k, _, _ in stack):
                        issues.append((decl_lineno, CHECK_NESTED_STRUCT, f"struct '{name}' defined inside a class body"))

                    stack.append((kind, name, depth))
                    pending = None
                else:
                    stack.append((None, None, depth))
            elif ch == '}':
                if stack and (stack[-1][2] == depth):
                    stack.pop()

                depth -= 1


def check_raw_output(lines, issues):
    for lineno, raw_line in enumerate(lines, start=1):
        line = strip_line_comment(raw_line)
        m    = RAW_OUTPUT.search(line)

        if m:
            issues.append((lineno, CHECK_RAW_OUTPUT, f"'{m.group(0)}' bypasses the central Logger (use LogInfo/LogWarning/LogError)"))


def check_pointer_naming(lines, issues):
    for lineno, raw_line in enumerate(lines, start=1):
        line = strip_line_comment(raw_line)

        for m in POINTER_DECL.finditer(line):
            if not looks_like_type(m.group('type')):
                continue

            name = m.group('name')

            if is_valid_pointer_name(name, m.group('stars')):
                continue

            expected = expected_pointer_prefix(name, m.group('stars')) + name[0].upper() + name[1:]
            issues.append((lineno, CHECK_POINTER_NAMING, f"pointer '{name}' should probably be '{expected}' (p/m_p prefix + PascalCase)"))


def mask_strings_and_casts(text):
    """Blank out (with spaces, preserving length/offsets) string/char literals and the
    <...> template-argument span of the four cast keywords, so neither can be mistaken
    for parens/&&/||/comparison operators during scanning."""
    chars = list(text)
    i     = 0
    n     = len(text)

    while i < n:
        ch = text[i]

        if ch in ('"', "'"):
            j = i + 1

            while (j < n) and (text[j] != ch):
                j += 2 if text[j] == '\\' else 1

            for k in range(i, min(j + 1, n)):
                chars[k] = ' '

            i = j + 1
            continue

        cast_kw = next((kw for kw in CAST_KEYWORDS
                         if text.startswith(kw, i) and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_'))),
                        None)

        if cast_kw is not None:
            j = i + len(cast_kw)

            while (j < n) and text[j] in ' \t':
                j += 1

            if (j < n) and (text[j] == '<'):
                depth = 1
                j += 1

                while (j < n) and (depth > 0):
                    if text[j] == '<':
                        depth += 1
                    elif text[j] == '>':
                        depth -= 1

                    j += 1

                for k in range(i, j):
                    chars[k] = ' '

                i = j
                continue

        i += 1

    return ''.join(chars)


def line_for_offset(line_starts, offset):
    lo, hi = 0, len(line_starts) - 1

    while lo < hi:
        mid = (lo + hi + 1) // 2

        if line_starts[mid] <= offset:
            lo = mid
        else:
            hi = mid - 1

    return lo + 1


def find_condition_span(masked_text, open_paren_offset):
    depth = 0
    i     = open_paren_offset

    while i < len(masked_text):
        if masked_text[i] == '(':
            depth += 1
        elif masked_text[i] == ')':
            depth -= 1

            if depth == 0:
                return open_paren_offset + 1, i

        i += 1

    return None


def split_top_level_operands(masked_text, start, end):
    operands = []
    depth    = 0
    op_start = start
    i        = start

    while i < end:
        ch = masked_text[i]

        if ch == '(':
            depth += 1
            i += 1
        elif ch == ')':
            depth -= 1
            i += 1
        elif (depth == 0) and masked_text.startswith('&&', i):
            operands.append((op_start, i))
            i += 2
            op_start = i
        elif (depth == 0) and masked_text.startswith('||', i):
            operands.append((op_start, i))
            i += 2
            op_start = i
        else:
            i += 1

    operands.append((op_start, end))

    return operands


def has_top_level_comparison(masked_operand):
    depth = 0
    i     = 0
    n     = len(masked_operand)

    while i < n:
        ch = masked_operand[i]

        if ch == '(':
            depth += 1
            i += 1
            continue

        if ch == ')':
            depth -= 1
            i += 1
            continue

        if depth == 0:
            if any(masked_operand.startswith(tok, i) for tok in NON_COMPARISON_OPS):
                i += 2
                continue

            op = next((op for op in COMPARISON_OPS if masked_operand.startswith(op, i)), None)

            if op is not None:
                return True

        i += 1

    return False


def find_compound_conditions(masked_text):
    """Yield (cond_start, cond_end, operands) for every if(...)/while(...) condition
    in masked_text that has 2+ top-level &&/|| operands. `operands` is a list of
    (start, end) offsets into masked_text (and, since masking preserves length, into
    the original text too)."""
    conditions = []

    for m in IF_WHILE.finditer(masked_text):
        span = find_condition_span(masked_text, m.end() - 1)

        if span is None:
            continue

        cond_start, cond_end = span
        operands = split_top_level_operands(masked_text, cond_start, cond_end)

        if len(operands) >= 2:
            conditions.append((cond_start, cond_end, operands))

    return conditions


def rfind_non_whitespace(text, end):
    """Return one past the last non-whitespace character in text[:end]."""
    i = end

    while (i > 0) and (text[i - 1] in ' \t\r\n'):
        i -= 1

    return i


def skip_whitespace(text, start):
    i = start

    while (i < len(text)) and (text[i] in ' \t\r\n'):
        i += 1

    return i


def is_fully_wrapped(text):
    text = text.strip()

    if not text.startswith('(') or not text.endswith(')'):
        return False

    depth = 0

    for i, ch in enumerate(text):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1

            if (depth == 0) and (i != len(text) - 1):
                return False

    return True


def check_compound_boolean_parens(text, issues):
    masked      = mask_strings_and_casts(text)
    line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == '\n']

    for _, _, operands in find_compound_conditions(masked):
        for op_start, op_end in operands:
            masked_operand = masked[op_start:op_end]

            if not has_top_level_comparison(masked_operand):
                continue

            original_operand = text[op_start:op_end]

            if is_fully_wrapped(original_operand):
                continue

            lineno = line_for_offset(line_starts, op_start)
            issues.append((lineno, CHECK_COMPOUND_BOOLEAN,
                            f"'{original_operand.strip()}' has a comparison but isn't fully parenthesized "
                            f"as its own operand"))


def check_compound_boolean_layout(text, issues):
    masked      = mask_strings_and_casts(text)
    line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == '\n']

    for cond_start, cond_end, operands in find_compound_conditions(masked):
        if '\n' not in text[cond_start:cond_end]:
            continue

        for idx in range(1, len(operands)):
            prev_end  = operands[idx - 1][1]
            operator  = text[prev_end:prev_end + 2]

            content_end = rfind_non_whitespace(text, prev_end)

            if '\n' in text[content_end:prev_end]:
                lineno = line_for_offset(line_starts, prev_end)
                issues.append((lineno, CHECK_COMPOUND_BOOLEAN_LAYOUT,
                                f"'{operator}' should trail the end of the previous operand's line, "
                                f"not lead its own"))
                continue

            gap_start     = prev_end + 2
            content_start = skip_whitespace(text, gap_start)

            if '\n' not in text[gap_start:content_start]:
                lineno = line_for_offset(line_starts, prev_end)
                issues.append((lineno, CHECK_COMPOUND_BOOLEAN_LAYOUT,
                                f"operand after '{operator}' should start on its own line"))


def check_file(path):
    with open(path) as f:
        lines = f.readlines()

    issues = []
    check_anonymous_namespaces(lines, issues)
    check_defaulted_functions(lines, issues)
    check_direct_brace_init(lines, issues)
    check_nested_structs(lines, issues)
    check_pointer_naming(lines, issues)
    check_raw_output(lines, issues)

    joined_text = ''.join(strip_line_comment_keep_newline(line) for line in lines)
    check_compound_boolean_parens(joined_text, issues)
    check_compound_boolean_layout(joined_text, issues)

    return sorted(issues)


def relative_to_repo(path, root):
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_ignore_entries(ignore_file):
    if not ignore_file.exists():
        return []

    with open(ignore_file) as f:
        data = json.load(f)

    entries = data.get('ignore', [])

    for entry in entries:
        if 'reason' not in entry:
            raise ValueError(f"{ignore_file}: ignore entry for {entry.get('file')} is missing a required 'reason'")

    return entries


def find_matching_entry(entries, used_entries, rel_path, check_id, lineno, message):
    for entry in entries:
        if id(entry) in used_entries:
            continue

        if entry.get('check') != check_id or entry.get('file') != rel_path:
            continue

        if ('line' in entry) and (entry['line'] != lineno):
            continue

        if ('identifier' in entry) and (entry['identifier'] not in message):
            continue

        return entry

    return None


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Heuristic checker for a handful of house C++ coding-standard rules '
                    '(see this file\'s module docstring for the full rule list and rationale).')
    subparsers = parser.add_subparsers(dest='command', required=True)

    check_parser = subparsers.add_parser('check', help='Scan files for coding-standard violations')
    check_parser.add_argument('paths', nargs='+',
                               help='Files or directories to scan')
    check_parser.add_argument('--root', type=Path, default=None,
                               help='Repo root used to resolve paths in the ignore config, and to locate the '
                                    'default ignore file (default: current working directory)')
    check_parser.add_argument('--config', '--ignore-file', dest='ignore_file', type=Path, default=None,
                               help='Path to the ignore-config JSON file '
                                    '(default: <root>/.check_coding_standards.json)')
    check_parser.add_argument('--ext', action='append', dest='extensions', default=None, metavar='PATTERN',
                               help='Glob pattern for files to scan when a directory is given; repeatable '
                                    '(default: --ext "*.h" --ext "*.cpp")')

    verify_parser = subparsers.add_parser(
        'verify-config',
        help='Validate the ignore-config file itself, independent of any scan')
    verify_parser.add_argument('--root', type=Path, default=None,
                                help='Repo root used to resolve entries\' "file" paths and to locate the default '
                                     'ignore file (default: current working directory)')
    verify_parser.add_argument('--config', '--ignore-file', dest='ignore_file', type=Path, default=None,
                                help='Path to the ignore-config JSON file '
                                     '(default: <root>/.check_coding_standards.json)')

    return parser


def collect_paths(args_paths, extensions):
    paths = []

    for arg in args_paths:
        p = Path(arg)

        if p.is_dir():
            for pattern in extensions:
                paths.extend(sorted(p.rglob(pattern)))
        else:
            paths.append(p)

    return paths


def run_checks(paths, root, ignore_file):
    ignore_entries = load_ignore_entries(ignore_file)
    used_entries   = set()
    total          = 0
    suppressed     = 0

    for path in paths:
        rel_path = relative_to_repo(path, root)

        for lineno, check_id, message in check_file(path):
            entry = find_matching_entry(ignore_entries, used_entries, rel_path, check_id, lineno, message)

            if entry is not None:
                used_entries.add(id(entry))
                suppressed += 1
                print(f"{path}:{lineno}: [suppressed: {check_id}] {message} -- {entry['reason']}")
                continue

            total += 1
            print(f"{path}:{lineno}: [{check_id}] {message}")

    for entry in ignore_entries:
        if id(entry) not in used_entries:
            print(f"{ignore_file}: warning: unused ignore entry for {entry.get('file')} ({entry.get('check')}) -- "
                  f"nothing matched it, consider removing it")

    if total == 0 and suppressed == 0:
        print("No issues found.")
        return 0

    print(f"\n{total} potential issue(s) found, {suppressed} suppressed (see {ignore_file.name}).")
    return 1 if total > 0 else 0


def verify_config(root, ignore_file):
    """Validate the ignore file on its own terms: well-formed JSON, every entry has the
    required fields and a known check id, its 'file' exists under root, and -- by actually
    re-running the checks against every referenced file that exists -- it still matches a
    real, current violation rather than one that's been fixed, renamed, or moved."""
    if not ignore_file.exists():
        print(f"{ignore_file}: does not exist (nothing to verify)")
        return 0

    try:
        data = json.loads(ignore_file.read_text())
    except json.JSONDecodeError as e:
        print(f"{ignore_file}: invalid JSON -- {e}")
        return 1

    if not isinstance(data, dict) or not isinstance(data.get('ignore'), list):
        print(f"{ignore_file}: expected a top-level object with an 'ignore' array")
        return 1

    entries   = data['ignore']
    problems  = []
    scannable = []

    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"entry {i}: not a JSON object")
            continue

        label   = entry.get('file', f'#{i}')
        missing = [field for field in REQUIRED_ENTRY_FIELDS if field not in entry]

        if missing:
            problems.append(f"entry {i} ({label}): missing required field(s): {', '.join(missing)}")
            continue

        if entry['check'] not in KNOWN_CHECKS:
            problems.append(f"entry {i} ({label}): unknown check id '{entry['check']}' "
                             f"(known: {', '.join(sorted(KNOWN_CHECKS))})")

        if ('line' not in entry) and ('identifier' not in entry):
            problems.append(f"entry {i} ({label}): has neither 'line' nor 'identifier', so it matches whichever "
                             f"occurrence of this check comes first in the file -- add one to pin it down")

        target = root / entry['file']

        if not target.is_file():
            problems.append(f"entry {i} ({label}): file does not exist at {target}")
            continue

        scannable.append((i, entry))

    used_entries  = set()
    files_to_scan = sorted({root / entry['file'] for _, entry in scannable})

    for path in files_to_scan:
        rel_path = relative_to_repo(path, root)

        for lineno, check_id, message in check_file(path):
            entry = find_matching_entry(entries, used_entries, rel_path, check_id, lineno, message)

            if entry is not None:
                used_entries.add(id(entry))

    for i, entry in scannable:
        if id(entry) not in used_entries:
            problems.append(f"entry {i} ({entry['file']}, {entry['check']}): doesn't match any current "
                             f"violation -- consider removing it")

    if problems:
        for p in problems:
            print(f"{ignore_file}: {p}")

        print(f"\n{len(problems)} problem(s) found in {ignore_file}.")
        return 1

    noun = 'entry' if len(entries) == 1 else 'entries'
    print(f"{ignore_file}: OK -- {len(entries)} {noun}, all well-formed, all files present, "
          f"all still matching a real violation.")
    return 0


def main():
    args = build_arg_parser().parse_args()

    root        = (args.root or Path.cwd()).resolve()
    ignore_file = (args.ignore_file or (root / '.check_coding_standards.json')).resolve()

    if args.command == 'verify-config':
        return verify_config(root, ignore_file)

    extensions = args.extensions or ['*.h', '*.cpp']
    paths      = collect_paths(args.paths, extensions)
    return run_checks(paths, root, ignore_file)


if __name__ == '__main__':
    sys.exit(main())
