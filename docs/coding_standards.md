# Coding Standards

## Purpose

This document captures the project-specific coding expectations that should guide new implementation work, reviews, and refactors.

## Language and Style

- Use C++17.
- Prefer straightforward, explicit code over abstract frameworks or clever metaprogramming.
- Keep implementation style consistent with the current codebase unless there is a strong reason to refactor a whole area.
- Avoid anonymous functions unless there is a significant advantage to using them.
- Avoid anonymous namespaces. It should either be typed as a project namespace or within the project namespace.

## Naming

- Types use `CamelCase`.
- Functions use `CamelCase`.
- Variables use `camelCase`.
- Class member variables use the `m_` prefix.
- Pointers as members use the `m_p` prefix, otherwise `p`.
- Prefer descriptive names over abbreviations unless the term is already standard in the domain.

## Braces and Layout

- Opening braces go on their own line.
- Keep indentation and surrounding file style consistent.
- if/else/while/for require {} even if they only have a single statement in them
- Do not reformat unrelated code when making local changes.

## Classes
- All variables must be init in the constructor, do not init in the body
- Do not use "default", require an implementation even if it is just {}
- Avoid defining structs inside classes
- Class constructors/methods should not have a space between the name and the (). MyClass().
- Align variables where posible. Types should be aligned, then variables.

## Functions
-- Align args and then their variables
-- Function name should be immediately followed by a `(`.
-- Each arg in the function should have its own line unless the entire function defintion can fit on a single line. The first args should always be on the same line as the function, though. Example:

void Foo(Type      arg0,
         Type2     arg1,
         LongType &arg2,
         Type3.   *pArg3)
{

}

## Conditionals

- Leave a space before condition parentheses: `if (...)`.
- For compound boolean expressions, parenthesize each comparison or logical unit.
- Example:

```cpp
if ((index >= 0) && (index < count) && (pointer != nullptr))
{
    // ...
}
```

## Initialization and Ownership

- Initialize class members in constructors rather than relying on scattered post-construction setup.
- For initialization, never use Type myVar{};, use Type myVar = {}; instead.
- Avoid introducing smart-pointer churn unless a subsystem is being intentionally refactored around ownership semantics.

## Data Modeling

- Prefer dedicated structs for grouped data such as coordinates or render instance payloads.
- Keep domain data simple and explicit.
- Avoid hiding important state transitions behind generic containers or opaque helper abstractions.
- Do not define structs in the body of a class

## Comments

- Add comments when the code would otherwise be difficult to parse or when domain intent is non-obvious.
- Do not add comments that merely restate the line below them.

## Error Handling

- Return explicit success/failure where the current codebase already follows that pattern.
- Preserve or improve existing error messages when adding new failure paths.
- Do not swallow Vulkan, file IO, or export failures silently, prefer to use a logger with severity levels

## Documentation Expectations

- Update `README.md` when user-visible behavior changes.
- Update design docs when architecture or exported behavior changes materially.
- Keep planning notes separate from coding standards.

## Variable types
- Prefer defined-size types like uint32_t, int64_t, etc.