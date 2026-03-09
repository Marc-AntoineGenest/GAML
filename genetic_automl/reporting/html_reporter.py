"""
HTML Report Generator — one self-contained HTML file per run.

Includes:
  - Run metadata (config, timing)
  - Fitness curve chart (Chart.js via CDN)
  - Best chromosome details
  - Full generation table
  - All evaluated chromosomes table
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from genetic_automl.config import PipelineConfig
from genetic_automl.genetic.chromosome import Chromosome
from genetic_automl.genetic.engine import EvolutionHistory
from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)


class HTMLReporter:
    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def generate(
        self,
        config: PipelineConfig,
        history: EvolutionHistory,
        final_test_score: Optional[float] = None,
        final_metric_name: Optional[str] = None,
        preprocessing_summary: Optional[Dict[str, Any]] = None,
        diversity_summary: Optional[Dict[str, Any]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        open_browser: bool = False,
    ) -> str:
        """Generate the HTML report and return its file path."""
        filename = f"run_{config.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(self.output_dir, filename)

        html = self._render(
            config=config,
            history=history,
            final_test_score=final_test_score,
            final_metric_name=final_metric_name,
            preprocessing_summary=preprocessing_summary or {},
            diversity_summary=diversity_summary or {},
            extra_info=extra_info or {},
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        log.info("HTML report saved: %s", filepath)

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(filepath)}")

        return filepath

    # ------------------------------------------------------------------

    def _render(
        self,
        config: PipelineConfig,
        history: EvolutionHistory,
        final_test_score: Optional[float],
        final_metric_name: Optional[str],
        preprocessing_summary: Dict[str, Any],
        diversity_summary: Dict[str, Any],
        extra_info: Dict[str, Any],
    ) -> str:
        best = history.best
        fitness_curve = json.dumps(history.fitness_curve())
        gen_labels = json.dumps([g.generation + 1 for g in history.generations])

        # Diversity data for JS charts
        hamming_json = json.dumps(diversity_summary.get('mean_hamming', []))
        mutrate_json = json.dumps(diversity_summary.get('mutation_rates', []))
        n_inject     = diversity_summary.get('n_injections_total', 0)
        n_boosts     = diversity_summary.get('n_boosts_total', 0)

        # Generation table rows
        gen_rows = "".join(
            f"""<tr>
                <td>{g.generation + 1}</td>
                <td>{g.best_fitness:.6f}</td>
                <td>{g.mean_fitness:.6f}</td>
                <td>{g.worst_fitness:.6f}</td>
                <td>{g.elapsed_seconds:.1f}s</td>
            </tr>"""
            for g in history.generations
        )

        # All chromosomes table (top 30)
        sorted_chroms = sorted(
            [c for c in history.all_chromosomes if c.fitness is not None],
            key=lambda c: c.fitness,
            reverse=True,
        )[:30]

        gene_keys = list(sorted_chroms[0].genes.keys()) if sorted_chroms else []
        gene_headers = "".join(f"<th>{k}</th>" for k in gene_keys)
        chrom_rows = "".join(
            f"""<tr>
                <td>{c.id}</td>
                <td>{c.generation}</td>
                <td class='{'fitness-best' if i == 0 else ''}'>{c.fitness:.6f}</td>
                {"".join(f"<td>{c.genes.get(k, '')}</td>" for k in gene_keys)}
            </tr>"""
            for i, c in enumerate(sorted_chroms)
        )

        best_genes_table = ""
        if best:
            best_genes_table = "".join(
                f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                for k, v in best.genes.items()
            )

        test_score_block = ""
        if final_test_score is not None:
            test_score_block = f"""
            <div class="metric-card highlight">
                <div class="metric-label">Final Test {final_metric_name or 'Score'}</div>
                <div class="metric-value">{final_test_score:.6f}</div>
            </div>"""

        # Preprocessing summary section
        pp_config = preprocessing_summary.get("config", {})
        pp_config_rows = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in pp_config.items()
        )
        dropped_feats = preprocessing_summary.get("correlation_dropped", [])
        selected_feats = preprocessing_summary.get("selected_features", [])
        pp_dropped_str = ", ".join(dropped_feats) if dropped_feats else "none"
        pp_selected_count = len(selected_feats) if selected_feats else "all"

        pp_section = f"""
  <h2>Best Preprocessing Config</h2>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="metric-label">Correlation-dropped Features</div>
      <div class="metric-value" style="font-size:1rem">{len(dropped_feats)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Features After Selection</div>
      <div class="metric-value" style="font-size:1rem">{pp_selected_count}</div>
    </div>
  </div>
  <div class="table-wrapper" style="max-height:260px">
    <table>
      <thead><tr><th>Preprocessing Step</th><th>Setting</th></tr></thead>
      <tbody>{pp_config_rows}</tbody>
    </table>
  </div>
  <p style="color:#64748b;font-size:0.8rem;margin-top:0.5rem">
    Dropped by correlation filter: <code>{pp_dropped_str}</code>
  </p>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoML Run — {config.run_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f1117; color: #e2e8f0; line-height: 1.6; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; color: #94a3b8; margin: 2rem 0 1rem; border-bottom: 1px solid #1e293b; padding-bottom: 0.5rem; }}
  .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; background: #1e40af; color: #bfdbfe; }}
  .meta {{ color: #64748b; font-size: 0.85rem; margin-top: 0.5rem; }}
  .metrics-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .metric-card {{ background: #1e293b; border-radius: 8px; padding: 1rem 1.5rem; min-width: 160px; border: 1px solid #334155; }}
  .metric-card.highlight {{ border-color: #3b82f6; background: #1e3a5f; }}
  .metric-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-value {{ font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-top: 0.25rem; }}
  .chart-container {{ background: #1e293b; border-radius: 8px; padding: 1.5rem; border: 1px solid #334155; margin: 1rem 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #1e293b; color: #94a3b8; padding: 0.6rem 0.8rem; text-align: left; font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05em; position: sticky; top: 0; }}
  td {{ padding: 0.5rem 0.8rem; border-bottom: 1px solid #1e293b; color: #cbd5e1; }}
  tr:hover td {{ background: #1a2638; }}
  .table-wrapper {{ background: #0d1421; border-radius: 8px; border: 1px solid #1e293b; overflow-x: auto; max-height: 400px; overflow-y: auto; }}
  .fitness-best {{ color: #34d399; font-weight: 700; }}
  .config-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.75rem; }}
  .config-item {{ background: #1e293b; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.85rem; }}
  .config-key {{ color: #64748b; font-size: 0.7rem; text-transform: uppercase; }}
  .config-val {{ color: #e2e8f0; font-weight: 500; margin-top: 0.1rem; }}
  footer {{ text-align: center; color: #334155; font-size: 0.75rem; margin-top: 3rem; padding: 1rem 0; }}
</style>
</head>
<body>
<div class="container">
  <h1>🧬 Genetic AutoML Run</h1>
  <div>
    <span class="badge">{config.problem_type.value}</span>
    <span class="badge" style="background:#065f46;color:#6ee7b7;margin-left:0.5rem;">{config.automl.backend}</span>
  </div>
  <p class="meta">
    Run ID: <code>{config.run_id}</code> &nbsp;|&nbsp;
    Name: <b>{config.run_name}</b> &nbsp;|&nbsp;
    Target: <code>{config.target_column}</code> &nbsp;|&nbsp;
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </p>

  <h2>Results</h2>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="metric-label">Best GA Fitness</div>
      <div class="metric-value">{f"{best.fitness:.6f}" if best else 'N/A'}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Generations Run</div>
      <div class="metric-value">{len(history.generations)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Individuals Evaluated</div>
      <div class="metric-value">{len(history.all_chromosomes)}</div>
    </div>
    {test_score_block}
  </div>

  <h2>Fitness Curve</h2>
  <div class="chart-container">
    <canvas id="fitnessChart" height="80"></canvas>
  </div>

  <h2>Best Configuration</h2>
  <div class="table-wrapper" style="max-height:200px">
    <table>
      <thead><tr><th>Gene</th><th>Value</th></tr></thead>
      <tbody>{best_genes_table}</tbody>
    </table>
  </div>

  {pp_section}

  <h2>Population Diversity</h2>
  <div class="metrics-row">
    <div class="metric-card">
      <div class="metric-label">Diversity Injections</div>
      <div class="metric-value" id="n-injections">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Mutation Boosts</div>
      <div class="metric-value" id="n-boosts">—</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Final Mean Hamming</div>
      <div class="metric-value" id="final-hamming">—</div>
    </div>
  </div>
  <canvas id="hamming-chart" style="max-height:220px;margin-bottom:2rem"></canvas>
  <canvas id="mutrate-chart" style="max-height:180px;margin-bottom:2rem"></canvas>

  <h2>Generation History</h2>
  <div class="table-wrapper">
    <table>
      <thead><tr><th>Gen</th><th>Best Fitness</th><th>Mean Fitness</th><th>Worst Fitness</th><th>Time</th></tr></thead>
      <tbody>{gen_rows}</tbody>
    </table>
  </div>

  <h2>Top 30 Individuals</h2>
  <div class="table-wrapper">
    <table>
      <thead><tr><th>ID</th><th>Gen</th><th>Fitness</th>{gene_headers}</tr></thead>
      <tbody>{chrom_rows}</tbody>
    </table>
  </div>

  <h2>Experiment Configuration</h2>
  <div class="config-grid">
    <div class="config-item"><div class="config-key">Population Size</div><div class="config-val">{config.genetic.population_size}</div></div>
    <div class="config-item"><div class="config-key">Generations</div><div class="config-val">{config.genetic.generations}</div></div>
    <div class="config-item"><div class="config-key">Mutation Rate</div><div class="config-val">{config.genetic.mutation_rate}</div></div>
    <div class="config-item"><div class="config-key">Crossover Rate</div><div class="config-val">{config.genetic.crossover_rate}</div></div>
    <div class="config-item"><div class="config-key">Elite Ratio</div><div class="config-val">{config.genetic.elite_ratio}</div></div>
    <div class="config-item"><div class="config-key">Tournament Size</div><div class="config-val">{config.genetic.tournament_size}</div></div>
    <div class="config-item"><div class="config-key">Early Stopping</div><div class="config-val">{config.genetic.early_stopping_rounds} rounds</div></div>
    <div class="config-item"><div class="config-key">Backend</div><div class="config-val">{config.automl.backend}</div></div>
  </div>

  <footer>Generated by Genetic AutoML Framework</footer>
</div>

<script>
// --- Diversity data ---
const hammingData  = {hamming_json};
const mutRateData  = {mutrate_json};
const nInjections  = {n_inject};
const nBoosts      = {n_boosts};
const finalHamming = hammingData.length ? hammingData[hammingData.length-1].toFixed(3) : 'n/a';
if (document.getElementById('n-injections')) document.getElementById('n-injections').textContent = nInjections;
if (document.getElementById('n-boosts'))     document.getElementById('n-boosts').textContent     = nBoosts;
if (document.getElementById('final-hamming'))document.getElementById('final-hamming').textContent = finalHamming;

if (document.getElementById('hamming-chart') && hammingData.length) {{
  new Chart(document.getElementById('hamming-chart').getContext('2d'), {{
    type: 'line',
    data: {{ labels: {gen_labels}, datasets: [{{ label: 'Mean Hamming Diversity', data: hammingData,
      borderColor: '#6366f1', backgroundColor: 'rgba(99,102,241,0.08)', fill: true, tension: 0.3, pointRadius: 4 }}] }},
    options: {{ plugins: {{ legend: {{ display: true }} }},
      scales: {{ y: {{ min: 0, max: 1, title: {{ display: true, text: 'Diversity (Hamming)' }} }} }} }}
  }});
}}
if (document.getElementById('mutrate-chart') && mutRateData.length) {{
  new Chart(document.getElementById('mutrate-chart').getContext('2d'), {{
    type: 'line',
    data: {{ labels: {gen_labels}, datasets: [{{ label: 'Adaptive Mutation Rate', data: mutRateData,
      borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.08)', fill: true, tension: 0.3, pointRadius: 4 }}] }},
    options: {{ plugins: {{ legend: {{ display: true }} }},
      scales: {{ y: {{ min: 0, max: 0.9, title: {{ display: true, text: 'Mutation Rate' }} }} }} }}
  }});
}}

// --- Fitness chart ---
const ctx = document.getElementById('fitnessChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: {gen_labels},
    datasets: [{{
      label: 'Best Fitness',
      data: {fitness_curve},
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.1)',
      borderWidth: 2,
      pointRadius: 4,
      pointBackgroundColor: '#3b82f6',
      fill: true,
      tension: 0.3,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8' }} }},
    }},
    scales: {{
      x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#64748b' }}, title: {{ display: true, text: 'Generation', color: '#64748b' }} }},
      y: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#64748b' }}, title: {{ display: true, text: 'Fitness', color: '#64748b' }} }},
    }}
  }}
}});
</script>
</body>
</html>"""
