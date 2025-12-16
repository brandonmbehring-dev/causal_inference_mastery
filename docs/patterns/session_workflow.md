# Session Workflow

Standard workflow for development sessions in causal_inference_mastery.

---

## Session Start

1. **Read CURRENT_WORK.md** - 30-second context resume
2. **Check for active plans** - `docs/plans/active/`
3. **Verify test status** - `pytest tests/ -m "not slow"` should pass

---

## CURRENT_WORK.md Format

```markdown
# Current Work

**Last Updated**: YYYY-MM-DD [Session N - Brief Title]

---

## Right Now

[STATUS]: Session N - Task Description

**Status**: [In Progress / Complete / Blocked]

**Session Summary**:
- Point 1
- Point 2

**Test Health**:
| Suite | Pass | Fail | Notes |
|-------|------|------|-------|
| Python | X | 0 | ... |
| Julia | Y | Z | ... |

**Next**: Session N+1 - Next task

---

## Session N-1 Summary (Date)
[Previous session notes...]
```

---

## Plan File Format (for tasks > 1 hour)

Location: `docs/plans/active/SESSION_N_TASK_YYYY-MM-DD_HH-MM.md`

```markdown
# Session N: Task Title

## Objective
[What we're trying to accomplish]

## Steps
1. Step 1
2. Step 2
3. ...

## Files to Modify
- `path/to/file1.py`
- `path/to/file2.jl`

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Notes
[Decisions made, alternatives considered]
```

---

## Session End

1. **Update CURRENT_WORK.md** with session summary
2. **Move completed plans** to `docs/plans/implemented/`
3. **Commit with standard format** (see below)
4. **Update session history** in CLAUDE.md if significant

---

## Git Commit Format

```
type(scope): Short description

- Detail 1
- Detail 2

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types
| Type | Use For |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `test` | Adding/fixing tests |
| `docs` | Documentation only |
| `refactor` | Code restructuring (no behavior change) |
| `validate` | Validation/Monte Carlo work |

### Scope Examples
- `rct`, `did`, `iv`, `rdd`, `psm`, `ipw`
- `cross-language`, `monte-carlo`
- `julia`, `python`

### Examples
```
feat(did): Add Callaway-Sant'Anna estimator

- Implemented group-time ATT estimation
- Added bootstrap SE for aggregated effects
- 12 new tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

```
fix(ipw): Handle perfect separation in propensity model

- Added detection for degenerate propensity scores
- Raise ValueError with diagnostic message
- Updated adversarial tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Typical Session Flow

```
1. Read CURRENT_WORK.md           (30 sec)
2. Run tests to verify baseline   (2 min)
3. Create plan if task > 1 hour   (10 min)
4. Implement with test-first      (varies)
5. Run full test suite            (5 min)
6. Update CURRENT_WORK.md         (2 min)
7. Commit with standard format    (1 min)
```

---

## Plan Lifecycle

```
docs/plans/active/     → Work in progress
docs/plans/implemented/ → Successfully completed
docs/archive/plans/    → Superseded or abandoned
```

When moving plans:
- **Implemented**: Plan was executed successfully
- **Archived**: Plan was abandoned or replaced by a better approach

---

*Last updated: 2025-12-16 (Session 37.5)*
