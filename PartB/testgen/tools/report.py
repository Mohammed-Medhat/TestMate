"""
TestMate — Report Generator
=============================
Generates results/report.html with bar charts for all 4 metrics
using embedded CSS/JS (no external dependencies).

Usage (from evaluate.py):
  from report import generate_report
  generate_report(all_results, output_dir="results")
"""

import os, json
from datetime import datetime


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TestMate — Evaluation Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0d1117; color: #e6edf3;
    padding: 2rem;
  }
  .header {
    text-align: center; margin-bottom: 2rem;
    border-bottom: 1px solid #30363d; padding-bottom: 1.5rem;
  }
  .header h1 { font-size: 2rem; color: #58a6ff; margin-bottom: 0.3rem; }
  .header .subtitle { color: #8b949e; font-size: 0.95rem; }
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem; margin-bottom: 2rem;
  }
  .summary-card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 1.2rem; text-align: center;
  }
  .summary-card .metric-value {
    font-size: 2rem; font-weight: 700; margin: 0.5rem 0;
  }
  .summary-card .metric-label { color: #8b949e; font-size: 0.85rem; }
  .pass { color: #3fb950; }
  .warn { color: #d29922; }
  .fail { color: #f85149; }
  .repo-section {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 1.5rem; margin-bottom: 1.5rem;
  }
  .repo-section h2 {
    font-size: 1.3rem; color: #58a6ff; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .metrics-row {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;
  }
  .metric-bar {
    background: #0d1117; border-radius: 6px; padding: 0.8rem;
  }
  .metric-bar .label { font-size: 0.8rem; color: #8b949e; margin-bottom: 0.4rem; }
  .metric-bar .bar-container {
    background: #21262d; border-radius: 4px; height: 24px;
    overflow: hidden; margin: 0.3rem 0;
  }
  .metric-bar .bar-fill {
    height: 100%; border-radius: 4px; transition: width 1s ease;
    display: flex; align-items: center; justify-content: flex-end;
    padding-right: 6px; font-size: 0.75rem; font-weight: 600;
    min-width: 30px;
  }
  .bar-green { background: linear-gradient(90deg, #238636, #3fb950); }
  .bar-yellow { background: linear-gradient(90deg, #9e6a03, #d29922); }
  .bar-red { background: linear-gradient(90deg, #b62324, #f85149); }
  .metric-bar .value {
    font-size: 1.4rem; font-weight: 700; margin-top: 0.2rem;
  }
  table {
    width: 100%; border-collapse: collapse; margin-top: 1rem;
  }
  th, td {
    padding: 0.7rem 1rem; text-align: left;
    border-bottom: 1px solid #21262d;
  }
  th { color: #8b949e; font-size: 0.85rem; text-transform: uppercase; }
  td { font-size: 0.95rem; }
  .footer {
    text-align: center; color: #484f58; font-size: 0.8rem;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #21262d;
  }
  .errors { color: #f85149; font-size: 0.85rem; margin-top: 0.5rem; }
</style>
</head>
<body>

<div class="header">
  <h1>🧪 TestMate — Evaluation Report</h1>
  <div class="subtitle">
    Qwen2.5-Coder-7B + LoRA · Graph-RAG Context · Self-Correcting Loop<br>
    Generated: {{generated_at}}
  </div>
</div>

<!-- Summary cards -->
<div class="summary-grid">
  <div class="summary-card">
    <div class="metric-label">Repos Evaluated</div>
    <div class="metric-value" style="color: #58a6ff;">{{num_repos}}</div>
  </div>
  <div class="summary-card">
    <div class="metric-label">Avg Pass Rate</div>
    <div class="metric-value {{pass_rate_class}}">{{avg_pass_rate}}%</div>
  </div>
  <div class="summary-card">
    <div class="metric-label">Avg Line Coverage</div>
    <div class="metric-value {{coverage_class}}">{{avg_line_coverage}}%</div>
  </div>
  <div class="summary-card">
    <div class="metric-label">Avg Mutation Score</div>
    <div class="metric-value {{mutation_class}}">{{avg_mutation_score}}%</div>
  </div>
</div>

<!-- Per-repo sections -->
{{repo_sections}}

<!-- Summary table -->
<div class="repo-section">
  <h2>📊 Comparison Table</h2>
  <table>
    <tr>
      <th>Repository</th>
      <th>Tests (Pass/Total)</th>
      <th>Pass Rate</th>
      <th>Line Coverage</th>
      <th>Branch Coverage</th>
      <th>Mutation Score</th>
    </tr>
    {{table_rows}}
  </table>
</div>

<div class="footer">
  TestMate · Autonomous Test Generation Agent · {{generated_at}}
</div>

</body>
</html>"""


REPO_SECTION_TEMPLATE = """
<div class="repo-section">
  <h2>📦 {{repo_name}}</h2>
  <div class="metrics-row">
    <div class="metric-bar">
      <div class="label">Test Pass Rate</div>
      <div class="bar-container">
        <div class="bar-fill {{pass_bar_class}}" style="width: {{pass_rate}}%">{{pass_rate}}%</div>
      </div>
      <div class="value {{pass_text_class}}">{{passed}}/{{total}} tests</div>
    </div>
    <div class="metric-bar">
      <div class="label">Line Coverage</div>
      <div class="bar-container">
        <div class="bar-fill {{line_bar_class}}" style="width: {{line_coverage}}%">{{line_coverage}}%</div>
      </div>
      <div class="value">{{line_coverage}}%</div>
    </div>
    <div class="metric-bar">
      <div class="label">Branch Coverage</div>
      <div class="bar-container">
        <div class="bar-fill {{branch_bar_class}}" style="width: {{branch_coverage}}%">{{branch_coverage}}%</div>
      </div>
      <div class="value">{{branch_coverage}}%</div>
    </div>
    <div class="metric-bar">
      <div class="label">Mutation Score</div>
      <div class="bar-container">
        <div class="bar-fill {{mut_bar_class}}" style="width: {{mutation_score}}%">{{mutation_score}}%</div>
      </div>
      <div class="value">{{mutation_score}}%</div>
    </div>
  </div>
  {{error_section}}
</div>
"""


def _bar_class(value: float) -> str:
    if value >= 80:
        return "bar-green"
    elif value >= 50:
        return "bar-yellow"
    return "bar-red"


def _text_class(value: float) -> str:
    if value >= 80:
        return "pass"
    elif value >= 50:
        return "warn"
    return "fail"


def generate_report(all_results: dict, output_dir: str = "results"):
    """
    Generate results/report.html from evaluation results.

    Args:
        all_results: {repo_name: {pass_rate, line_coverage, branch_coverage, mutation_score, ...}}
        output_dir: where to write report.html and per-repo JSONs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save per-repo JSON files
    for repo_name, metrics in all_results.items():
        json_path = os.path.join(output_dir, f"{repo_name}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"   💾 Saved: {json_path}")

    # Calculate averages
    repos = list(all_results.keys())
    num_repos = len(repos)

    if num_repos == 0:
        print("   ⚠️  No results to report")
        return

    avg_pass = sum(r.get("pass_rate", 0) for r in all_results.values()) / num_repos
    avg_line = sum(r.get("line_coverage", 0) for r in all_results.values()) / num_repos
    avg_branch = sum(r.get("branch_coverage", 0) for r in all_results.values()) / num_repos
    avg_mut = sum(r.get("mutation_score", 0) for r in all_results.values()) / num_repos

    # Build repo sections
    repo_sections = ""
    table_rows = ""

    for repo_name, m in all_results.items():
        pr = round(m.get("pass_rate", 0), 1)
        lc = round(m.get("line_coverage", 0), 1)
        bc = round(m.get("branch_coverage", 0), 1)
        ms = round(m.get("mutation_score", 0), 1)
        passed = m.get("passed_tests", 0)
        total = m.get("total_tests", 0)

        errors = m.get("errors", [])
        error_section = ""
        if errors:
            error_html = "<br>".join(f"• {e}" for e in errors[:5])
            error_section = f'<div class="errors">⚠️ {error_html}</div>'

        section = REPO_SECTION_TEMPLATE
        section = section.replace("{{repo_name}}", repo_name)
        section = section.replace("{{pass_rate}}", str(pr))
        section = section.replace("{{line_coverage}}", str(lc))
        section = section.replace("{{branch_coverage}}", str(bc))
        section = section.replace("{{mutation_score}}", str(ms))
        section = section.replace("{{passed}}", str(passed))
        section = section.replace("{{total}}", str(total))
        section = section.replace("{{pass_bar_class}}", _bar_class(pr))
        section = section.replace("{{line_bar_class}}", _bar_class(lc))
        section = section.replace("{{branch_bar_class}}", _bar_class(bc))
        section = section.replace("{{mut_bar_class}}", _bar_class(ms))
        section = section.replace("{{pass_text_class}}", _text_class(pr))
        section = section.replace("{{error_section}}", error_section)
        repo_sections += section

        table_rows += f"""
    <tr>
      <td>{repo_name}</td>
      <td>{passed}/{total}</td>
      <td class="{_text_class(pr)}">{pr}%</td>
      <td class="{_text_class(lc)}">{lc}%</td>
      <td class="{_text_class(bc)}">{bc}%</td>
      <td class="{_text_class(ms)}">{ms}%</td>
    </tr>"""

    # Build final HTML
    html = HTML_TEMPLATE
    html = html.replace("{{generated_at}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("{{num_repos}}", str(num_repos))
    html = html.replace("{{avg_pass_rate}}", str(round(avg_pass, 1)))
    html = html.replace("{{avg_line_coverage}}", str(round(avg_line, 1)))
    html = html.replace("{{avg_mutation_score}}", str(round(avg_mut, 1)))
    html = html.replace("{{pass_rate_class}}", _text_class(avg_pass))
    html = html.replace("{{coverage_class}}", _text_class(avg_line))
    html = html.replace("{{mutation_class}}", _text_class(avg_mut))
    html = html.replace("{{repo_sections}}", repo_sections)
    html = html.replace("{{table_rows}}", table_rows)

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n   📊 Report saved: {report_path}")
    return report_path
