# AGENTS.md

## General

- Prefer the smallest possible change that solves the task.
- Do not refactor unrelated code.
- Do not modify files unrelated to the requested task.
- Preserve the existing architecture and coding style.
- Prefer simple implementations over abstractions.

## Testing

- Do not add unit tests unless explicitly requested.
- Do not add smoke tests unless explicitly requested.
- Do not run tests unless explicitly requested.
- Do not run linters, formatters, or type checkers unless explicitly requested.

## Validation

- Avoid unnecessary validation and defensive programming.
- Do not add extra assertions, try/except blocks, or fallback logic unless needed for the requested behavior.
- Assume existing internal inputs are valid unless the task says otherwise.

## Workflow

- Do not create extra documentation.
- Do not create new abstractions for one-off functionality.
- Do not inspect large parts of the repository when the relevant files are already known.
- Stop once the requested change is implemented.

## Response

- Keep the final response short.
- State only what was changed and which files were modified.