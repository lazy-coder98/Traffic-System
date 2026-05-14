    const API = "/api";
    const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

    let lineChart, barChart, donutChart, comparisonChart, rewardChart;

    function updateHourLabel(v) {
      document.getElementById('p-hour-val').textContent = v;
      const h = parseInt(v);
      const ampm = h >= 12 ? 'PM' : 'AM';
      const disp = h === 0 ? '12:00 AM' : h < 12 ? `${h}:00 AM` : h === 12 ? '12:00 PM' : `${h - 12}:00 PM`;
      document.getElementById('hour-label').textContent = disp;
    }

    function handleHourChange(v) {
      updateHourLabel(v);
      markSelectedHeatmapCell();
    }

    function handlePredictionInputChange() {
      syncSelectedJunction();
      loadHeatmap();
    }

    // ── Fetch stats & populate cards ──────────────────────
    async function loadStats() {
      try {
        const r = await fetch(`${API}/stats`);
        const d = await r.json();

        document.getElementById('s-records').textContent = d.total_records.toLocaleString();
        document.getElementById('s-avg').textContent = d.avg_vehicles;
        document.getElementById('s-peak').textContent = d.max_vehicles;
        document.getElementById('s-acc').textContent = (d.model_metrics.rfc_accuracy * 100).toFixed(1) + '%';
        document.getElementById('m-records').textContent = d.total_records.toLocaleString() + ' rows';
        document.getElementById('m-period').textContent = d.date_range.start + ' to ' + d.date_range.end;

        document.getElementById('lr-rmse').textContent = d.model_metrics.lr.rmse;
        document.getElementById('lr-r2').textContent = d.model_metrics.lr.r2;
        document.getElementById('rf-rmse').textContent = d.model_metrics.rf.rmse;
        document.getElementById('rf-r2').textContent = d.model_metrics.rf.r2;

        // Donut
        const dist = d.congestion_dist;
        const total = (dist.Low || 0) + (dist.Medium || 0) + (dist.High || 0);
        const colors = { Low: '#42c983', Medium: '#f4b740', High: '#ef6a67' };
        const keys = ['Low', 'Medium', 'High'];

        const legend = document.getElementById('donut-legend');
        legend.innerHTML = keys.map(k => `
      <div class="legend-item">
        <div class="legend-dot" style="background:${colors[k]}"></div>
        <div>
          <div style="font-size:14px;font-weight:600">${k}</div>
          <div style="font-size:11px;color:var(--muted);font-family:var(--mono)">${((dist[k] || 0) / total * 100).toFixed(1)}% · ${(dist[k] || 0).toLocaleString()}</div>
        </div>
      </div>`).join('');

        if (donutChart) donutChart.destroy();
        donutChart = new Chart(document.getElementById('donutChart'), {
          type: 'doughnut',
          data: {
            labels: keys,
            datasets: [{
              data: keys.map(k => dist[k] || 0),
              backgroundColor: keys.map(k => colors[k] + '33'),
              borderColor: keys.map(k => colors[k]),
              borderWidth: 2,
              hoverOffset: 4
            }]
          },
          options: {
            responsive: false,
            cutout: '72%',
            plugins: {
              legend: { display: false }, tooltip: {
                callbacks: {
                  label: ctx => ` ${ctx.label}: ${ctx.raw.toLocaleString()}`
                }
              }
            },
            animation: { duration: 800, easing: 'easeInOutQuart' }
          }
        });
      } catch (e) { console.error('Stats error', e); }
    }

    // ── Hourly trend ──────────────────────────────────────
    async function loadJunctionTrend() {
      const j = document.getElementById('junc-select').value;
      try {
        const r = await fetch(`${API}/junction_trend?junction=${j}`);
        const d = await r.json();
        const labels = d.map(x => x.hour + ':00');
        const vals = d.map(x => x.avg_vehicles);

        if (lineChart) lineChart.destroy();
        lineChart = new Chart(document.getElementById('lineChart'), {
          type: 'line',
          data: {
            labels,
            datasets: [{
              label: 'Avg Vehicles',
              data: vals,
              borderColor: '#39c6b8',
              backgroundColor: 'rgba(57,198,184,.12)',
              tension: 0.4,
              pointRadius: 3,
              pointBackgroundColor: '#39c6b8',
              fill: true
            }]
          },
          options: chartOpts('Avg Vehicles / hr')
        });
      } catch (e) { console.error(e); }
    }

    // ── Daily pattern ─────────────────────────────────────
    async function loadDailyPattern() {
      try {
        const r = await fetch(`${API}/daily_pattern`);
        const d = await r.json();

        if (barChart) barChart.destroy();
        barChart = new Chart(document.getElementById('barChart'), {
          type: 'bar',
          data: {
            labels: d.map(x => x.day),
            datasets: [{
              label: 'Avg Vehicles',
              data: d.map(x => x.avg_vehicles),
              backgroundColor: d.map((_, i) =>
                i >= 5 ? 'rgba(244,183,64,.72)' : 'rgba(57,198,184,.58)'),
              borderColor: d.map((_, i) =>
                i >= 5 ? '#f4b740' : '#39c6b8'),
              borderWidth: 1,
              borderRadius: 5
            }]
          },
          options: chartOpts('Avg Vehicles / day')
        });
      } catch (e) { console.error(e); }
    }

    // ── Heatmap ───────────────────────────────────────────
    async function loadHeatmap() {
      try {
        const selectedJunction = document.getElementById('p-junction').value;
        const r = await fetch(`${API}/heatmap?junction=${selectedJunction}`);
        const d = await r.json();

        // build lookup
        const lookup = {};
        let maxV = 0;
        d.forEach(c => {
          lookup[`${c.hour}-${c.day}`] = c.vehicles;
          if (c.vehicles > maxV) maxV = c.vehicles;
        });

        const grid = document.getElementById('heatmap-grid');
        grid.innerHTML = '';

        // header row
        grid.appendChild(makeEl('div', 'hm-header', ''));
        DAYS.forEach(day => {
          const el = makeEl('div', 'hm-header', day);
          grid.appendChild(el);
        });

        for (let h = 0; h < 24; h++) {
          const lbl = makeEl('div', 'hm-label', h + 'h');
          grid.appendChild(lbl);
          for (let day = 0; day < 7; day++) {
            const v = lookup[`${h}-${day}`] || 0;
            const cell = document.createElement('div');
            cell.className = 'hm-cell';
            const level = congestionLevel(v);
            cell.dataset.hour = h;
            cell.dataset.day = day;
            cell.dataset.level = level;
            cell.style.background = heatColor(v);
            cell.title = `Junction ${selectedJunction} / ${DAYS[day]} ${h}:00 - ${v} vehicles (${level})`;
            grid.appendChild(cell);
          }
        }
        markSelectedHeatmapCell();
      } catch (e) { console.error(e); }
    }

    function makeEl(tag, cls, text) {
      const el = document.createElement(tag);
      el.className = cls;
      el.textContent = text;
      return el;
    }

    function syncSelectedJunction() {
      const selected = document.getElementById('p-junction').value;
      document.querySelectorAll('.junction-node').forEach(node => {
        node.classList.toggle('active', node.dataset.junction === selected);
      });
    }

    function markSelectedHeatmapCell() {
      const hour = document.getElementById('p-hour').value;
      const day = document.getElementById('p-day').value;
      document.querySelectorAll('.hm-cell').forEach(cell => {
        cell.classList.toggle('selected', cell.dataset.hour === hour && cell.dataset.day === day);
      });
    }

    function congestionLevel(v) {
      if (v < 35) return 'Low';
      if (v < 55) return 'Medium';
      return 'High';
    }

    function heatColor(v) {
      const level = congestionLevel(v);
      if (level === 'Low') return 'rgba(66,201,131,.72)';
      if (level === 'Medium') return 'rgba(244,183,64,.76)';
      return 'rgba(239,106,103,.8)';
    }

    // ── Predict ───────────────────────────────────────────
    async function predict() {
      const btn = document.getElementById('predict-btn');
      btn.disabled = true;
      btn.textContent = 'Predicting…';
      const payload = {
        junction: document.getElementById('p-junction').value,
        hour: document.getElementById('p-hour').value,
        day: document.getElementById('p-day').value
      };
      try {
        const r = await fetch(`${API}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await r.json();
        document.getElementById('r-vehicles').textContent = d.predicted_vehicles + ' vehicles';
        document.getElementById('r-signal').textContent = d.signal_time_seconds + 's (rule)';
        const cEl = document.getElementById('r-congestion');
        cEl.innerHTML = `<span class="badge ${d.congestion_level}">${d.congestion_level}</span>`;
        const activeNode = document.querySelector(`.junction-node[data-junction="${payload.junction}"]`);
        if (activeNode) {
          activeNode.classList.remove('medium', 'high');
          if (d.congestion_level === 'Medium') activeNode.classList.add('medium');
          if (d.congestion_level === 'High') activeNode.classList.add('high');
        }
        document.getElementById('result-box').classList.add('visible');

        // RL recommendation display
        if (d.rl_recommendation) {
          const rl = d.rl_recommendation;
          document.getElementById('r-rl-signal').textContent = rl.optimal_signal_seconds + 's';
          document.getElementById('r-rl-action').innerHTML = `<span class="junc-action ${rl.management_action}">${rl.action_label}</span>`;
          document.getElementById('r-rl-priority').textContent = rl.priority_score + ' / 100';
          const pa = rl.phase_allocation;
          const totalCycle = pa.cycle_seconds;
          const gPct = (rl.optimal_signal_seconds / totalCycle * 100).toFixed(0);
          const aPct = (pa.amber_seconds / totalCycle * 100).toFixed(0);
          const rPct = Math.max(0, 100 - gPct - aPct);
          document.getElementById('r-phase-bar').innerHTML =
            `<div class="green-phase" style="width:${gPct}%"></div><div class="amber-phase" style="width:${aPct}%"></div><div class="red-phase" style="width:${rPct}%"></div>`;
          document.getElementById('r-phase-label').textContent =
            `G ${rl.optimal_signal_seconds}s · A ${pa.amber_seconds}s · R ${pa.red_seconds}s (cycle ${pa.cycle_seconds}s)`;
          document.getElementById('rl-result-row').style.display = 'grid';
        }
      } catch (e) {
        alert('Backend not reachable. Please try again in a moment.');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Now';
      }
    }

    // ── Shared chart options ──────────────────────────────
    function chartOpts(yLabel) {
      return {
        responsive: true,
        animation: { duration: 600 },
        plugins: {
          legend: { display: false }, tooltip: {
            backgroundColor: '#151d26',
            borderColor: '#263241',
            borderWidth: 1,
            titleFont: { family: 'Space Mono', size: 11 },
            bodyFont: { family: 'Space Mono', size: 12 }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(238,244,248,.06)' }, ticks: { color: '#93a4b5', font: { family: 'Space Mono', size: 10 } } },
          y: { grid: { color: 'rgba(238,244,248,.06)' }, ticks: { color: '#93a4b5', font: { family: 'Space Mono', size: 10 } }, title: { display: false } }
        }
      };
    }

    // ── Management: Command Center ───────────────────────
    async function loadManagementStatus() {
      try {
        const r = await fetch(`${API}/management/status`);
        const d = await r.json();
        const tm = d.training_metrics;
        document.getElementById('rl-episodes').textContent = tm.episodes;
        document.getElementById('rl-converged').textContent = 'Ep ' + tm.converged_at;
        document.getElementById('rl-reward').textContent = tm.avg_reward_last_50;

        const center = document.getElementById('cmd-center');
        const juncNames = ['City Core', 'Market Road', 'Ring Road', 'Bypass'];
        center.innerHTML = d.junctions.map(j => {
          const cl = j.congestion_level.toLowerCase();
          return `<div class="cmd-junction">
        <div class="junc-id ${cl}">J${j.junction}</div>
        <div class="junc-info">
          <div class="junc-name">Junction ${j.junction} — ${juncNames[j.junction - 1]}</div>
          <div class="junc-meta">${j.predicted_vehicles} veh · ${j.optimal_signal_seconds}s signal · P${j.priority_score}</div>
        </div>
        <div class="junc-action ${j.management_action}">${j.management_action}</div>
      </div>`;
        }).join('');

        // Load reward chart
        loadRewardChart(tm.reward_history_sample);
      } catch (e) { console.error('Mgmt status error', e); }
    }

    // ── Management: Comparison ───────────────────────────
    async function loadComparison() {
      try {
        const r = await fetch(`${API}/management/comparison`);
        const d = await r.json();
        document.getElementById('rl-efficiency').textContent = d.overall.signal_efficiency_pct + '%';

        const labels = [];
        const rlData = [];
        const ruleData = [];
        d.comparison.forEach(j => {
          j.periods.forEach(p => {
            labels.push(`J${j.junction} ${p.period.substring(0, 3)}`);
            rlData.push(p.avg_rl_signal);
            ruleData.push(p.avg_rule_signal);
          });
        });

        if (comparisonChart) comparisonChart.destroy();
        comparisonChart = new Chart(document.getElementById('comparisonChart'), {
          type: 'bar',
          data: {
            labels,
            datasets: [
              { label: 'RL Policy', data: rlData, backgroundColor: 'rgba(57,198,184,.65)', borderColor: '#39c6b8', borderWidth: 1, borderRadius: 4 },
              { label: 'Rule-Based', data: ruleData, backgroundColor: 'rgba(239,106,103,.45)', borderColor: '#ef6a67', borderWidth: 1, borderRadius: 4 }
            ]
          },
          options: {
            ...chartOpts('Signal (s)'),
            plugins: {
              legend: { display: true, position: 'top', labels: { color: '#93a4b5', font: { family: 'Space Mono', size: 10 }, boxWidth: 12, padding: 12 } },
              tooltip: { backgroundColor: '#151d26', borderColor: '#263241', borderWidth: 1, titleFont: { family: 'Space Mono', size: 11 }, bodyFont: { family: 'Space Mono', size: 12 } }
            }
          }
        });
      } catch (e) { console.error('Comparison error', e); }
    }

    // ── Management: Policy Heatmap ───────────────────────
    async function loadPolicyHeatmap() {
      try {
        const j = document.getElementById('policy-junc-select').value;
        const r = await fetch(`${API}/management/policy_heatmap?junction=${j}`);
        const d = await r.json();

        const lookup = {};
        d.forEach(c => { lookup[`${c.hour}-${c.day}`] = c; });

        const grid = document.getElementById('policy-heatmap-grid');
        grid.innerHTML = '';

        grid.appendChild(makeEl('div', 'hm-header', ''));
        DAYS.forEach(day => grid.appendChild(makeEl('div', 'hm-header', day)));

        for (let h = 0; h < 24; h++) {
          grid.appendChild(makeEl('div', 'hm-label', h + 'h'));
          for (let day = 0; day < 7; day++) {
            const c = lookup[`${h}-${day}`] || { rl_signal: 25, management_action: 'Normal', predicted_vehicles: 0 };
            const cell = document.createElement('div');
            cell.className = 'hm-cell';
            cell.style.background = policyHeatColor(c.rl_signal);
            cell.title = `J${j} ${DAYS[day]} ${h}:00 — RL: ${c.rl_signal}s · Rule: ${c.rule_signal}s · ${c.management_action} · ${c.predicted_vehicles} veh`;
            grid.appendChild(cell);
          }
        }
      } catch (e) { console.error('Policy heatmap error', e); }
    }

    function policyHeatColor(signal) {
      if (signal <= 15) return 'rgba(57,198,184,.45)';
      if (signal <= 25) return 'rgba(57,198,184,.75)';
      if (signal <= 35) return 'rgba(66,201,131,.65)';
      if (signal <= 50) return 'rgba(244,183,64,.65)';
      return 'rgba(239,106,103,.7)';
    }

    // ── Management: Reward Chart ─────────────────────────
    function loadRewardChart(rewardSample) {
      if (rewardChart) rewardChart.destroy();
      const labels = rewardSample.map((_, i) => i * Math.round(800 / rewardSample.length));
      rewardChart = new Chart(document.getElementById('rewardChart'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Episode Reward',
            data: rewardSample,
            borderColor: '#39c6b8',
            backgroundColor: 'rgba(57,198,184,.1)',
            tension: 0.3,
            pointRadius: 1,
            fill: true,
            borderWidth: 2
          }]
        },
        options: {
          ...chartOpts('Reward'),
          plugins: {
            legend: { display: false },
            tooltip: { backgroundColor: '#151d26', borderColor: '#263241', borderWidth: 1, titleFont: { family: 'Space Mono', size: 11 }, bodyFont: { family: 'Space Mono', size: 12 } }
          }
        }
      });
    }

    // ── Init & Auto-Refresh ──────────────────────────────
    async function initAll() {
      await Promise.all([
        loadStats(),
        loadJunctionTrend(),
        loadDailyPattern(),
        loadHeatmap(),
        loadManagementStatus(),
        loadComparison(),
        loadPolicyHeatmap()
      ]);
      syncSelectedJunction();
    }
    
    // Initial load
    initAll();
    
    // Auto-refresh every 60 seconds
    setInterval(() => {
      console.log('Auto-refreshing dashboard data...');
      initAll();
    }, 60000);
