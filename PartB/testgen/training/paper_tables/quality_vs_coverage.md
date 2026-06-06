# Coverage vs Correctness vs Bug-Detection

The three axes that matter — and why a single coverage number is misleading. Coverage = did inputs run the code; pass@1 = are the assertions correct; mutation = would the test *catch a bug*.

| variant / mode | line cov% | pass@1% | mutation% | what it shows |
|---|---|---|---|---|
| TestMate (suite) | 91.6 | 17.6 | 74.1 | coverage-optimized |
| TestMate (quality) | 42.5 | 16.7 | - | correctness-optimized |

**Headline:** run with `--mutation` to fill the bug-detection column and apply the decision rule. Until then, the proven hook is the coverage–correctness gap (91% coverage, ~18% correct assertions).
