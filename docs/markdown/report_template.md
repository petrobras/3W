<header class="report-header">
  <div class="header-overlay"></div> <div class="header-content">
    <img src="/home/pedro/Workspace/UFRJ/3W/3WToolkit/docs/figures/3w_logo.png" alt="Logo" class="header-logo">
    <h1>{{ title }}</h1>
    <p class="author-info">
      <strong>Autor:</strong> {{ author }} | <strong>Gerado em:</strong> {{ generation_date }}
    </p>
  </div>
</header>

<link rel="stylesheet" href="/home/pedro/Workspace/UFRJ/3W/3WToolkit/docs/markdown/petro.css">


<main>

<div class="info-box principio">
    <div class="info-box-header">Sumário</div>
    <div class="info-box-content">
      <ol>
        <li><a href="#performance-evaluation">Performance Evaluation</a></li>
        <li><a href="#model-overview">Model Overview</a></li>
        <li><a href="#visualizations">Visualizations</a></li>
      </ol>
    </div>
  </div>

<!-- <div class="toc">
  <h2>Table of Contents</h2>
  <ol>
    <li><a href="#performance-evaluation">Performance Evaluation</a></li>
    <li><a href="#model-overview">Model Overview</a></li>
    <li><a href="#visualizations">Visualizations</a></li>
  </ol>
</div> -->

<details open class="info-box explicacao">
<summary class="info-box-header">
  Performance Evaluation
</summary>
  <div class="info-box-content">
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Métrica</th>
            <th>Valor</th>
          </tr>
        </thead>
        <tbody>
          {% for name, value in calculated_metrics.items() -%}
          <tr>
            <td>{{ name.replace("get_", "").replace("_", " ").title() }}</td>
            <td>{{ "%.4f"|format(value) }}</td>
          </tr>
          {%- endfor %}
        </tbody>
      </table>
    </div>
  </div>
</details>

<details open class="info-box explicacao">
<summary class="info-box-header">Model Overview
</summary>
  <div class="info-box-content">
      <h3>Model and Data</h3>
      <h4>Model Configuration</h4>
      <ul>
        <li><strong>Type:</strong> <code>{{ model_type }}</code></li>
        <li><strong>Parameters:</strong>
          <ul>
            {% for param, value in model_config.items() %}
              <li>{{ param.replace("_", " ").title() }}: <code>{{ value }}</code></li>
            {% endfor %}
          </ul>
        </li>
      </ul>
      <h4>Data Split</h4>
      <ul>
        <li><strong>Training Samples:</strong> {{ train_samples }}</li>
        <li><strong>Test Samples:</strong> {{ test_samples }}</li>
      </ul>
  </div>
</details>

<details open class="info-box explicacao">
<summary class="info-box-header">Visualizations
</summary>

<div class="info-box-content">
<div class="visualization-grid">
  {% for param, value in plot_data.items() %}
    <div class="viz-card">
      <div class="card-header">
        <h3>{{ value.title }}</h3>
      </div>
      <div class="card-body">
        <img src="{{ value.img_path }}" alt="{{ value.alt }}">
      </div>
    </div>
  {% endfor %}
</div>
</div>
</details>

</main>
