/* Price history plus a volatility envelope.
   The deployed models classify direction only and output no price, so nothing
   here is a predicted price. The forward band is +/- 1.28 standard deviations
   of recent realised volatility, scaled by the square root of the horizon: a
   description of how far this market has lately tended to move, drawn to give
   the direction probability a sense of scale. */
(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const store = window.CryptoLens;
  const Z80 = 1.2816;                        // two-sided 80% under a normal assumption
  const HISTORY_SHARE = 0.62;                // fraction of the plot given to the past
  const view = {rows: [], generation: 0, points: [], hover: null, geometry: null};
  const money = value => value == null || !Number.isFinite(Number(value)) ? '—'
    : '$' + Number(value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  const compact = value => '$' + Math.round(value).toLocaleString('en-US');
  const percent = value => value == null ? 'Unavailable' : (Number(value) * 100).toFixed(1) + '%';
  const token = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const alpha = (hex, a) => {
    const value = hex.replace('#', '');
    if (value.length !== 6) return hex;
    const [r, g, b] = [0, 2, 4].map(i => parseInt(value.slice(i, i + 2), 16));
    return `rgba(${r},${g},${b},${a})`;
  };

  const wrap = document.querySelector('.market-chart-wrap');
  const tooltip = document.createElement('div');
  tooltip.className = 'chart-tooltip';
  tooltip.hidden = true;
  wrap.append(tooltip);

  async function json(url) {
    const response = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(20000)});
    if (!response.ok) throw new Error('API returned ' + response.status);
    return response.json();
  }

  function envelope() {
    const market = store.context?.context?.market;
    const price = Number(market?.price);
    const sigma = Number(market?.volatility_24h);
    if (!Number.isFinite(price) || !Number.isFinite(sigma) || sigma <= 0) return null;
    return {price, sigma, hours: store.horizon, half: Z80 * sigma * Math.sqrt(store.horizon)};
  }

  function describe() {
    const env = envelope();
    const hours = String(store.horizon);
    byId('envelope-horizon').textContent = hours;
    byId('envelope-horizon-2').textContent = hours;
    byId('price-now').textContent = money(store.context?.context?.market?.price);
    byId('envelope-range').textContent = env
      ? `${compact(env.price * (1 - env.half))} – ${compact(env.price * (1 + env.half))}`
      : '—';
    byId('envelope-sigma').textContent = env ? (env.sigma * 100).toFixed(2) + '% per hour' : '—';
  }

  function draw() {
    const canvas = byId('market-price-chart');
    const box = canvas.getBoundingClientRect();
    if (!box.width) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = box.width * ratio;
    canvas.height = box.height * ratio;
    const ctx = canvas.getContext('2d');
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, box.width, box.height);

    const points = [...view.rows].reverse().filter(row => Number.isFinite(Number(row.current_price))).slice(-48);
    view.points = points;
    byId('market-chart-empty').hidden = points.length > 0;
    if (!points.length) { view.geometry = null; return; }

    const prices = points.map(row => Number(row.current_price));
    const env = envelope();
    const anchor = env ? env.price : prices[prices.length - 1];
    let min = Math.min(...prices, anchor), max = Math.max(...prices, anchor);
    if (env) {
      min = Math.min(min, anchor * (1 - env.half));
      max = Math.max(max, anchor * (1 + env.half));
    }
    const pad = Math.max((max - min) * .12, max * .0008);
    min -= pad; max += pad;

    const left = 12, right = box.width - 58, top = 20, bottom = box.height - 36;
    const nowX = left + (right - left) * (env ? HISTORY_SHARE : 1);
    const x = index => left + (nowX - left) * (points.length <= 1 ? 1 : index / (points.length - 1));
    const y = price => bottom - (price - min) / (max - min || 1) * (bottom - top);
    view.geometry = {x, y, left, right, top, bottom, prices};

    const grid = token('--border') || '#e7e9f1';
    const muted = token('--muted') || '#8b95a7';
    const accent = token('--cyan') || '#6267e8';

    // forward region, visually separated so the past is never confused with it
    if (env) {
      ctx.fillStyle = alpha(accent, .05);
      ctx.fillRect(nowX, top, right - nowX, bottom - top);
    }

    ctx.font = '10px DM Sans, sans-serif';
    ctx.strokeStyle = grid; ctx.fillStyle = muted; ctx.lineWidth = 1;
    for (let index = 0; index < 4; index++) {
      const value = max - (max - min) * index / 3, lineY = y(value);
      ctx.beginPath(); ctx.setLineDash([3, 5]); ctx.moveTo(left, lineY); ctx.lineTo(right, lineY); ctx.stroke();
      ctx.fillText(compact(value), right + 9, lineY + 3);
    }
    ctx.setLineDash([]);

    if (env) {
      /* The flat line is the model's classification boundary: it asks only
         whether price closes above or below the current one. So the envelope is
         split there and each side shaded by that side's probability. The band's
         height is realised volatility; the split is the model's actual output. */
      const forecast = store.context?.context?.forecasts?.[String(store.horizon)];
      const up = Number(forecast?.probability_up);
      const state = forecast?.signal_state;
      const qualified = ['UP', 'DOWN'].includes(state);
      const upTone = qualified ? (token('--green') || '#238a64') : muted;
      const downTone = qualified ? (token('--red') || '#c85265') : muted;

      const steps = 36;
      const edge = sign => {
        const path = [];
        for (let i = 0; i <= steps; i++) {
          const fraction = i / steps;
          const half = Z80 * env.sigma * Math.sqrt(env.hours * fraction);
          path.push([nowX + (right - nowX) * fraction, y(anchor * (1 + sign * half))]);
        }
        return path;
      };
      const region = (path, tone, probability) => {
        ctx.beginPath();
        ctx.moveTo(nowX, y(anchor));
        path.forEach(([px, py]) => ctx.lineTo(px, py));
        ctx.lineTo(right, y(anchor));
        ctx.closePath();
        // a dominant probability reads as a visibly stronger region
        ctx.fillStyle = alpha(tone, Number.isFinite(probability) ? .07 + .3 * probability : .12);
        ctx.fill();
      };
      region(edge(1), upTone, up);
      region(edge(-1), downTone, Number.isFinite(up) ? 1 - up : NaN);

      if (Number.isFinite(up)) {
        ctx.font = '600 11px DM Sans, sans-serif';
        ctx.textAlign = 'right';
        const halfEnd = Z80 * env.sigma * Math.sqrt(env.hours);
        ctx.fillStyle = upTone;
        ctx.fillText(`UP ${(up * 100).toFixed(0)}%`, right - 6, y(anchor * (1 + halfEnd * .55)));
        ctx.fillStyle = downTone;
        ctx.fillText(`DOWN ${((1 - up) * 100).toFixed(0)}%`, right - 6, y(anchor * (1 - halfEnd * .55)) + 4);
        ctx.textAlign = 'start';
        ctx.font = '10px DM Sans, sans-serif';
      }

      // the decision boundary itself: no drift is predicted, so it stays flat
      ctx.strokeStyle = muted; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(nowX, y(anchor)); ctx.lineTo(right, y(anchor)); ctx.stroke();
      ctx.setLineDash([]);
    }

    // history: area then line
    const area = ctx.createLinearGradient(0, top, 0, bottom);
    area.addColorStop(0, alpha(accent, .16));
    area.addColorStop(1, alpha(accent, 0));
    ctx.beginPath();
    points.forEach((row, index) => index ? ctx.lineTo(x(index), y(prices[index])) : ctx.moveTo(x(index), y(prices[index])));
    ctx.lineTo(x(points.length - 1), bottom); ctx.lineTo(x(0), bottom); ctx.closePath();
    ctx.fillStyle = area; ctx.fill();

    ctx.strokeStyle = accent; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
    points.forEach((row, index) => index ? ctx.lineTo(x(index), y(prices[index])) : ctx.moveTo(x(index), y(prices[index])));
    ctx.stroke();
    ctx.fillStyle = accent;
    ctx.beginPath(); ctx.arc(x(points.length - 1), y(prices[prices.length - 1]), 4, 0, Math.PI * 2); ctx.fill();

    if (env) {
      ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(nowX, top); ctx.lineTo(nowX, bottom); ctx.stroke(); ctx.setLineDash([]);
    }

    if (view.hover != null && points[view.hover]) {
      ctx.strokeStyle = muted; ctx.lineWidth = 1; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(x(view.hover), top); ctx.lineTo(x(view.hover), bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(x(view.hover), y(prices[view.hover]), 4.5, 0, Math.PI * 2); ctx.stroke();
    }

    ctx.fillStyle = muted; ctx.textAlign = 'center';
    const span = points.length;
    [0, Math.round((span - 1) / 2), span - 1].filter((v, i, a) => a.indexOf(v) === i).forEach(index => {
      ctx.fillText(index === span - 1 ? 'Now' : `−${span - 1 - index}h`, x(index), box.height - 10);
    });
    if (env) ctx.fillText(`+${env.hours}h`, right, box.height - 10);
    ctx.textAlign = 'start';
    canvas.setAttribute('aria-label', chartLabel(prices, env));
  }

  function chartLabel(prices, env) {
    const forecast = store.context?.context?.forecasts?.[String(store.horizon)];
    const base = `Price history from ${money(prices[0])} to ${money(prices[prices.length - 1])}. `;
    const band = env
      ? `An 80% volatility envelope over the next ${env.hours} hours spans ${compact(env.price * (1 - env.half))} to ${compact(env.price * (1 + env.half))}. This is realised volatility, not a predicted price. `
      : '';
    return base + band + (forecast
      ? `The ${store.horizon} hour probability of up is ${percent(forecast.probability_up)}.`
      : 'Forecast unavailable.');
  }

  function moveTooltip(event) {
    const box = byId('market-price-chart').getBoundingClientRect();
    if (!view.geometry || !view.points.length) return;
    const offsetX = event.clientX - box.left;
    let nearest = 0, best = Infinity;
    view.points.forEach((row, index) => {
      const distance = Math.abs(view.geometry.x(index) - offsetX);
      if (distance < best) { best = distance; nearest = index; }
    });
    if (view.hover === nearest) return;
    view.hover = nearest;
    const row = view.points[nearest];
    tooltip.hidden = false;
    tooltip.replaceChildren();
    const price = document.createElement('strong');
    price.textContent = money(row.current_price);
    const stamp = document.createElement('span');
    stamp.textContent = new Date(row.issued_at).toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
    tooltip.append(price, stamp);
    tooltip.style.left = Math.min(Math.max(view.geometry.x(nearest), 62), box.width - 62) + 'px';
    draw();
  }

  function clearTooltip() {
    if (view.hover == null) return;
    view.hover = null; tooltip.hidden = true; draw();
  }

  async function loadHistory() {
    if (!store.deployed) return;
    const generation = ++view.generation;
    try {
      const history = await json('/monitoring/history?horizon=' + store.horizon + '&days=30&limit=200');
      if (generation !== view.generation) return;
      view.rows = history.items || [];
      draw();
    } catch {
      if (generation !== view.generation) return;
      byId('market-chart-note').textContent = 'Price history unavailable · retrying with the dashboard refresh';
    }
  }

  const canvas = byId('market-price-chart');
  canvas.addEventListener('mousemove', moveTooltip);
  canvas.addEventListener('mouseleave', clearTooltip);
  window.addEventListener('resize', () => { clearTooltip(); draw(); });
  window.addEventListener('cryptolens:view', draw);
  store.on('context', () => { describe(); draw(); });
  store.on('horizon', () => { describe(); loadHistory(); draw(); });
  store.on('asset', () => { view.rows = []; clearTooltip(); describe(); loadHistory(); draw(); });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) loadHistory(); });
  describe();
  loadHistory();
  setInterval(() => { if (!document.hidden) loadHistory(); }, 30000);
})();
