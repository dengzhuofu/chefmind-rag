"""
评估报告生成器
从评估结果 JSON 生成独立 HTML 报告页面

使用方法:
    python generate_report.py evaluation_results/20260605_102540/summary.json
    python generate_report.py evaluation_results/20260605_102540/summary.json -o my_report.html
"""

import argparse
import json
import html
from pathlib import Path


def generate_html(data: dict) -> str:
    """从评估结果数据生成 HTML 报告"""

    total = data.get("total_tests", 0)
    passed = data.get("passed", 0)
    failed = data.get("failed", 0)
    pass_rate = data.get("pass_rate", 0)
    custom = data.get("custom_metrics", {})
    ragas = data.get("ragas_metrics", {})
    by_route = data.get("by_route_type", {})
    by_diff = data.get("by_difficulty", {})
    results = data.get("test_results", [])

    # 按路由类型准备图表数据
    route_labels = json.dumps(list(by_route.keys()), ensure_ascii=False)
    route_rates = [round(v["passed"] / v["total"] * 100, 1) if v["total"] > 0 else 0 for v in by_route.values()]
    route_rates_json = json.dumps(route_rates)

    # 按难度准备图表数据
    diff_labels = json.dumps(list(by_diff.keys()), ensure_ascii=False)
    diff_rates = [round(v["passed"] / v["total"] * 100, 1) if v["total"] > 0 else 0 for v in by_diff.values()]
    diff_rates_json = json.dumps(diff_rates)

    # 指标卡片
    recall = custom.get("recall_at_5", 0)
    citation = custom.get("citation_accuracy", 0)
    refuse = custom.get("refuse_accuracy", 0)

    # RAGAS 指标
    ragas_html = ""
    if ragas:
        for k, v in ragas.items():
            if v is not None:
                ragas_html += f'<div class="metric-item"><span class="metric-label">{k}</span><span class="metric-value">{v:.4f}</span></div>'
            else:
                ragas_html += f'<div class="metric-item"><span class="metric-label">{k}</span><span class="metric-value na">N/A</span></div>'

    # 测试用例表格行
    rows_html = ""
    for r in results:
        tc_id = html.escape(r.get("id", ""))
        query = html.escape(r.get("query", ""))
        passed_flag = r.get("passed", False)
        metrics = r.get("metrics", {})
        errors = r.get("errors", [])
        status_class = "pass" if passed_flag else "fail"
        status_text = "PASS" if passed_flag else "FAIL"
        recall_v = metrics.get("recall_at_5", 0)
        citation_v = metrics.get("citation_accuracy", 0)
        refuse_v = metrics.get("refuse_accuracy", 0)
        errors_text = html.escape("; ".join(errors)) if errors else "-"

        rows_html += f"""
        <tr class="{status_class}">
            <td>{tc_id}</td>
            <td class="query-cell">{query}</td>
            <td><span class="badge {status_class}">{status_text}</span></td>
            <td>{recall_v:.2f}</td>
            <td>{citation_v:.2f}</td>
            <td>{refuse_v:.2f}</td>
            <td class="error-cell">{errors_text}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ChefMind RAG Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f2f5; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.subtitle {{ color: #666; margin-bottom: 32px; font-size: 14px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
.card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); text-align: center; }}
.card-label {{ font-size: 13px; color: #888; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
.card-value {{ font-size: 36px; font-weight: 700; }}
.card-value.green {{ color: #22c55e; }}
.card-value.yellow {{ color: #eab308; }}
.card-value.red {{ color: #ef4444; }}
.charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 32px; }}
@media (max-width: 768px) {{ .charts {{ grid-template-columns: 1fr; }} }}
.chart-box {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.chart-box h3 {{ font-size: 16px; margin-bottom: 16px; color: #333; }}
.ragas-section {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 32px; }}
.ragas-section h3 {{ font-size: 16px; margin-bottom: 16px; color: #333; }}
.ragas-grid {{ display: flex; flex-wrap: wrap; gap: 24px; }}
.metric-item {{ display: flex; flex-direction: column; min-width: 140px; }}
.metric-label {{ font-size: 12px; color: #888; text-transform: uppercase; margin-bottom: 4px; }}
.metric-value {{ font-size: 20px; font-weight: 600; color: #333; }}
.metric-value.na {{ color: #ccc; }}
.table-section {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.table-section h3 {{ font-size: 16px; margin-bottom: 16px; color: #333; }}
.table-controls {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
.filter-btn {{ padding: 6px 14px; border: 1px solid #ddd; border-radius: 6px; background: #fff; cursor: pointer; font-size: 13px; transition: all 0.2s; }}
.filter-btn:hover {{ border-color: #3b82f6; color: #3b82f6; }}
.filter-btn.active {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th {{ text-align: left; padding: 10px 12px; border-bottom: 2px solid #e5e7eb; color: #666; font-weight: 600; font-size: 12px; text-transform: uppercase; cursor: pointer; user-select: none; white-space: nowrap; }}
th:hover {{ color: #3b82f6; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }}
tr.fail {{ background: #fef2f2; }}
tr.pass {{ background: #fff; }}
tr:hover {{ background: #f8fafc; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
.badge.pass {{ background: #dcfce7; color: #16a34a; }}
.badge.fail {{ background: #fee2e2; color: #dc2626; }}
.query-cell {{ max-width: 280px; }}
.error-cell {{ max-width: 200px; color: #dc2626; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
    <h1>ChefMind RAG Evaluation Report</h1>
    <p class="subtitle">Generated from evaluation results &middot; {total} test cases</p>

    <div class="cards">
        <div class="card">
            <div class="card-label">Pass Rate</div>
            <div class="card-value {"green" if pass_rate >= 0.8 else "yellow" if pass_rate >= 0.6 else "red"}">{pass_rate:.1%}</div>
        </div>
        <div class="card">
            <div class="card-label">Recall@5</div>
            <div class="card-value {"green" if recall >= 0.8 else "yellow" if recall >= 0.5 else "red"}">{recall:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">Citation Accuracy</div>
            <div class="card-value {"green" if citation >= 0.8 else "yellow" if citation >= 0.5 else "red"}">{citation:.4f}</div>
        </div>
        <div class="card">
            <div class="card-label">Refuse Accuracy</div>
            <div class="card-value {"green" if refuse >= 0.8 else "yellow" if refuse >= 0.5 else "red"}">{refuse:.4f}</div>
        </div>
    </div>

    <div class="charts">
        <div class="chart-box">
            <h3>Pass Rate by Route Type</h3>
            <canvas id="routeChart"></canvas>
        </div>
        <div class="chart-box">
            <h3>Pass Rate by Difficulty</h3>
            <canvas id="diffChart"></canvas>
        </div>
    </div>

    {"<div class='ragas-section'><h3>RAGAS Metrics</h3><div class='ragas-grid'>" + ragas_html + "</div></div>" if ragas_html else ""}

    <div class="table-section">
        <h3>Test Results ({total} cases)</h3>
        <div class="table-controls">
            <button class="filter-btn active" onclick="filterRows('all',this)">All ({total})</button>
            <button class="filter-btn" onclick="filterRows('fail',this)">Failed ({failed})</button>
            <button class="filter-btn" onclick="filterRows('pass',this)">Passed ({passed})</button>
        </div>
        <table id="resultsTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">ID</th>
                    <th onclick="sortTable(1)">Query</th>
                    <th onclick="sortTable(2)">Status</th>
                    <th onclick="sortTable(3)">Recall@5</th>
                    <th onclick="sortTable(4)">Citation</th>
                    <th onclick="sortTable(5)">Refuse</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</div>

<script>
new Chart(document.getElementById('routeChart'), {{
    type: 'bar',
    data: {{
        labels: {route_labels},
        datasets: [{{
            label: 'Pass Rate (%)',
            data: {route_rates_json},
            backgroundColor: {route_rates_json}.map(v => v >= 80 ? '#22c55e' : v >= 50 ? '#eab308' : '#ef4444'),
            borderRadius: 6,
            borderSkipped: false,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            y: {{ beginAtZero: true, max: 100, ticks: {{ callback: v => v + '%' }} }},
            x: {{ ticks: {{ font: {{ size: 11 }} }} }}
        }}
    }}
}});

new Chart(document.getElementById('diffChart'), {{
    type: 'doughnut',
    data: {{
        labels: {diff_labels},
        datasets: [{{
            data: {diff_rates_json},
            backgroundColor: ['#22c55e', '#eab308', '#ef4444', '#3b82f6'],
            borderWidth: 0,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{
            legend: {{ position: 'bottom' }},
            tooltip: {{ callbacks: {{ label: ctx => ctx.label + ': ' + ctx.parsed + '%' }} }}
        }}
    }}
}});

function filterRows(type, btn) {{
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('#resultsTable tbody tr').forEach(tr => {{
        tr.style.display = (type === 'all') ? '' : tr.classList.contains(type) ? '' : 'none';
    }});
}}

let sortDir = {{}};
function sortTable(col) {{
    const tbody = document.querySelector('#resultsTable tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    sortDir[col] = !sortDir[col];
    rows.sort((a, b) => {{
        let va = a.cells[col].textContent.trim();
        let vb = b.cells[col].textContent.trim();
        const na = parseFloat(va), nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) {{ va = na; vb = nb; }}
        if (va < vb) return sortDir[col] ? -1 : 1;
        if (va > vb) return sortDir[col] ? 1 : -1;
        return 0;
    }});
    rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation HTML report")
    parser.add_argument("input", help="Path to summary.json")
    parser.add_argument("-o", "--output", help="Output HTML path (default: same dir as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    data = json.loads(input_path.read_text(encoding="utf-8"))
    html_content = generate_html(data)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "report.html"

    output_path.write_text(html_content, encoding="utf-8")
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
