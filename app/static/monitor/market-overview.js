(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const view = {asset: 'BTC', horizon: 1, rows: [], context: null, generation: 0};
  const assets = {
    BTC: {name: 'Bitcoin', pair: 'BTC / USDT', configured: true},
    ETH: {name: 'Ethereum', pair: 'ETH / USDT', configured: false},
    ADA: {name: 'Cardano', pair: 'ADA / USDT', configured: false}
  };
  const money = value => value == null ? '—' : '$' + Number(value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  const percent = value => value == null ? 'Unavailable' : (Number(value) * 100).toFixed(1) + '%';

  async function json(url) {
    const response = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(20000)});
    if (!response.ok) throw new Error('API returned ' + response.status);
    return response.json();
  }

  function setSelection(selector, active) {
    document.querySelectorAll(selector).forEach(button => {
      const selected = button === active;
      button.classList.toggle(selector.includes('asset') ? 'active' : 'selected', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }

  function renderAsset() {
    const asset = assets[view.asset];
    byId('market-forecast-symbol').textContent = asset.pair;
    byId('market-forecast-title').textContent = asset.name + ' directional forecast';
    const empty = byId('market-chart-empty');
    if (!asset.configured) {
      empty.hidden = false;
      byId('market-chart-price').textContent = '—';
      byId('market-chart-direction').textContent = 'Model unavailable';
      byId('market-chart-probability').textContent = 'No validated forecast pipeline';
      byId('market-forecast-description').textContent = 'This asset is ready in the interface, but no trained production model is deployed.';
      byId('market-chart-empty-title').textContent = asset.name + ' forecast not configured';
      draw();
      return;
    }
    empty.hidden = true;
    byId('market-forecast-description').textContent = "Saved reference prices with the selected model's directional probability.";
    const forecast = view.context?.context?.forecasts?.[String(view.horizon)];
    byId('market-chart-price').textContent = money(view.context?.context?.market?.price);
    byId('market-chart-direction').textContent = forecast ? (forecast.signal_state === 'UP' || forecast.signal_state === 'DOWN' ? forecast.signal_state : 'No reliable signal') : 'Awaiting forecast';
    byId('market-chart-probability').textContent = forecast ? 'P(UP) ' + percent(forecast.probability_up) + ' · ' + view.horizon + 'h horizon' : 'Probability unavailable';
    draw();
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
    if (view.asset !== 'BTC' || !view.rows.length) return;
    const points = [...view.rows].reverse().filter(row => Number.isFinite(Number(row.current_price))).slice(-48);
    if (!points.length) return;
    const prices = points.map(row => Number(row.current_price));
    let min = Math.min(...prices), max = Math.max(...prices);
    const padding = Math.max((max - min) * .18, max * .001);
    min -= padding; max += padding;
    const left = 12, right = box.width - 58, top = 18, bottom = box.height - 35;
    const x = index => left + (right - left) * (points.length <= 1 ? .5 : index / (points.length - 1));
    const y = price => bottom - (price - min) / (max - min || 1) * (bottom - top);
    ctx.font = '10px DM Sans, sans-serif';
    ctx.strokeStyle = '#e7e9f1'; ctx.fillStyle = '#8b95a7'; ctx.lineWidth = 1;
    for (let index = 0; index < 4; index++) {
      const value = max - (max - min) * index / 3, lineY = y(value);
      ctx.beginPath(); ctx.setLineDash([3, 5]); ctx.moveTo(left, lineY); ctx.lineTo(box.width - 8, lineY); ctx.stroke();
      ctx.fillText('$' + Math.round(value).toLocaleString('en-US'), right + 9, lineY + 3);
    }
    ctx.setLineDash([]); ctx.strokeStyle = '#6267e8'; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.beginPath();
    points.forEach((row, index) => index ? ctx.lineTo(x(index), y(Number(row.current_price))) : ctx.moveTo(x(index), y(Number(row.current_price))));
    ctx.stroke();
    const finalX = x(points.length - 1), finalY = y(prices[prices.length - 1]);
    ctx.fillStyle = '#6267e8'; ctx.beginPath(); ctx.arc(finalX, finalY, 4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#9699ef'; ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(finalX, finalY); ctx.lineTo(box.width - 10, finalY); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = '#c9cbee'; ctx.lineWidth = 1; ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(finalX, top); ctx.lineTo(finalX, bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = '#6e7890'; ctx.textAlign = 'center';
    const ticks = [...new Set([0, Math.round((points.length - 1) / 2), points.length - 1])];
    ticks.forEach(index => ctx.fillText(index === points.length - 1 ? 'Now' : new Intl.DateTimeFormat('en-GB', {timeZone: 'UTC', month: 'short', day: 'numeric'}).format(new Date(points[index].issued_at)), x(index), box.height - 9));
    canvas.setAttribute('aria-label', assetChartLabel(points, prices));
  }

  function assetChartLabel(points, prices) {
    const forecast = view.context?.context?.forecasts?.[String(view.horizon)];
    return 'Bitcoin saved reference-price history from ' + money(prices[0]) + ' to ' + money(prices[prices.length - 1]) + '. ' + (forecast ? view.horizon + ' hour probability of up is ' + percent(forecast.probability_up) + '. No price target is generated.' : 'Forecast unavailable.');
  }

  async function load() {
    const generation = ++view.generation;
    try {
      const [context, history] = await Promise.all([
        json('/context'),
        json('/monitoring/history?horizon=' + view.horizon + '&days=30&limit=200')
      ]);
      if (generation !== view.generation) return;
      view.context = context; view.rows = history.items || [];
      const price = context.context?.market?.price;
      const momentum = context.context?.market?.momentum_24h;
      byId('asset-btc-price').textContent = money(price);
      byId('asset-btc-change').textContent = momentum == null ? '24h change unavailable' : (momentum >= 0 ? '↗ +' : '↘ ') + (momentum * 100).toFixed(2) + '% · 24h';
      byId('asset-btc-change').classList.toggle('negative', momentum < 0);
      renderAsset();
    } catch {
      byId('market-chart-direction').textContent = 'API unavailable';
      byId('market-chart-probability').textContent = 'Retrying with the dashboard refresh';
    }
  }

  document.querySelectorAll('[data-asset]').forEach(button => button.addEventListener('click', () => {
    view.asset = button.dataset.asset;
    setSelection('[data-asset]', button);
    renderAsset();
  }));
  document.querySelectorAll('[data-market-horizon]').forEach(button => button.addEventListener('click', () => {
    view.horizon = Number(button.dataset.marketHorizon);
    setSelection('[data-market-horizon]', button);
    const matching = document.querySelector('[data-horizon="' + view.horizon + '"]');
    if (matching && view.asset === 'BTC') matching.click();
    load();
  }));
  window.addEventListener('resize', draw);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) load(); });
  load();
  setInterval(() => { if (!document.hidden) load(); }, 30000);
})();
