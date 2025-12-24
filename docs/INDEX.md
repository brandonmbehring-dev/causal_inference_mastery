# Documentation Index

Single entry point for causal_inference_mastery documentation.

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [CURRENT_WORK.md](../CURRENT_WORK.md) | 30-second session context |
| [ROADMAP.md](ROADMAP.md) | Master plan, phase tracking |
| [METHODOLOGICAL_CONCERNS.md](METHODOLOGICAL_CONCERNS.md) | 13 tracked concerns |
| [METHOD_SELECTION.md](METHOD_SELECTION.md) | Decision tree for method selection |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Debug guide for validation issues |
| [GLOSSARY.md](GLOSSARY.md) | Terminology reference |
| [FAILURE_MODES.md](FAILURE_MODES.md) | Method failure taxonomy |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Commands cheat sheet |
| [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) | Test xfails & edge cases |

---

## Navigation

| I want to... | Go to... |
|--------------|----------|
| Resume current session | [../CURRENT_WORK.md](../CURRENT_WORK.md) |
| See project status | [ROADMAP.md](ROADMAP.md) |
| Choose a method | [METHOD_SELECTION.md](METHOD_SELECTION.md) |
| Debug a test failure | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Look up a term | [GLOSSARY.md](GLOSSARY.md) |
| Understand failure modes | [FAILURE_MODES.md](FAILURE_MODES.md) |
| Run tests | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| Understand validation | [patterns/validation.md](patterns/validation.md) |
| Check known issues | [METHODOLOGICAL_CONCERNS.md](METHODOLOGICAL_CONCERNS.md) |
| See test limitations | [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) |

---

## Session History

Session archives are organized in `docs/archive/sessions/`:

| Resource | Purpose |
|----------|---------|
| [Session Index](archive/sessions/INDEX.md) | Quick lookup by method/session |
| [SESSION_85-95](archive/sessions/) | Advanced methods (Selection, QTE, MTE, etc.) |
| [SESSION_63-82](archive/sessions/) | RKD, Bunching, Context Engineering |
| [SESSION_42-62](archive/sessions/) | CATE, Sensitivity, SCM |
| [SESSION_1-19](archive/sessions/) | Foundation (RCT, PSM, DiD, IV, RDD) |

---

## Directory Structure

```
docs/
├── INDEX.md                    # You are here
├── QUICK_REFERENCE.md          # Commands cheat sheet
├── ROADMAP.md                  # Master plan
├── GAP_ANALYSIS.md             # Missing methods (updated Session 96)
├── METHODOLOGICAL_CONCERNS.md  # 13 tracked concerns
├── METHOD_SELECTION.md         # Decision tree
├── TROUBLESHOOTING.md          # Debug guide
├── GLOSSARY.md                 # Terminology
├── FAILURE_MODES.md            # Failure taxonomy
├── KNOWN_LIMITATIONS.md        # Test xfails
├── patterns/                   # Reusable patterns
│   ├── validation.md           # 6-layer validation architecture
│   ├── testing.md              # Test-first workflow
│   └── session_workflow.md
├── plans/
│   ├── active/                 # In-progress plans
│   └── implemented/            # Completed plans
├── standards/
│   └── PHASE_COMPLETION_STANDARDS.md
├── checklists/
│   └── PHASE_COMPLETION_CHECKLIST.md
└── archive/
    └── sessions/               # Historical SESSION_*.md + INDEX.md
```

---

## Key Entry Points

### For Development
1. Check [../CURRENT_WORK.md](../CURRENT_WORK.md) for session context
2. Run tests with [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Check [METHODOLOGICAL_CONCERNS.md](METHODOLOGICAL_CONCERNS.md) before implementing

### For Understanding
1. [ROADMAP.md](ROADMAP.md) - Project overview and phases
2. [patterns/validation.md](patterns/validation.md) - Validation architecture
3. [../CLAUDE.md](../CLAUDE.md) - AI assistant guidance

---

*Last updated: 2025-12-23 (Session 96 - Documentation Architecture)*
