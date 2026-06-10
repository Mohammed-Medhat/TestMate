# Coverage vs Correctness vs Bug-Detection

The three axes that matter — and why a single coverage number is misleading. Coverage = did inputs run the code; pass@1 = are the assertions correct; mutation = would the test *catch a bug*.

| variant / mode | line cov% | pass@1% | mutation% | what it shows |
|---|---|---|---|---|
| TestMate (suite) | 93.8 | 97.6 | 81.5 | coverage-optimized |
| base (suite, no_lora) | 78.9 | 80.0 | 89.8 | raw model, suite |
| TestMate (quality) | 42.5 | 16.7 | - | correctness-optimized |

**Headline (data-supported):** the contribution is the *insight + system*, not beating the base per file (mutation 81.5% vs base 89.8% per file). TestMate produces bug-catching suites on **more programs** (205 vs 168); coverage is misleading (93.8% cov / 97.6% pass rate in suite mode, 42.5% cov / 16.7% pass rate in quality mode).
