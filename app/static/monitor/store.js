/* Shared state for the dashboard: one horizon selection, one /context poll.
   Modules subscribe instead of fetching the same endpoints independently. */
'use strict';
window.CryptoLens = (() => {
  const bus = new EventTarget();
  const state = {horizon: 1, asset: null, assets: [], prices: {}, holdout: null, predictions: null, context: null, contextError: false};
  let generation = 0;

  const emit = (name, detail) => bus.dispatchEvent(new CustomEvent(name, {detail}));
  const on = (name, handler) => bus.addEventListener(name, event => handler(event.detail));
  const current = () => state.assets.find(item => item.code === state.asset) || null;

  function setHorizon(value) {
    const horizon = Number(value);
    if (![1, 6, 24].includes(horizon) || horizon === state.horizon) return;
    state.horizon = horizon;
    document.querySelectorAll('[data-horizon]').forEach(button => {
      const active = Number(button.dataset.horizon) === horizon;
      button.classList.toggle('active', active);
      button.setAttribute('aria-pressed', String(active));
    });
    emit('horizon', horizon);
  }

  const json = async url => {
    const response = await fetch(url, {cache: 'no-store', signal: AbortSignal.timeout(20000)});
    if (!response.ok) throw new Error('API returned ' + response.status);
    return response.json();
  };

  /* The analyst router owns /context. When it is unavailable the core forecast
     endpoints still work, so the forecast view degrades instead of going blank.
     Reliability is absent on this path, so every horizon reports UNKNOWN: an
     unqualified direction must never be presented as a usable signal. */
  async function loadFallback() {
    const [forecasts, market] = await Promise.all([json('/forecasts'), json('/market')]);
    const entries = Object.entries(forecasts).map(([horizon, forecast]) => [horizon, {
      raw_direction: forecast.direction,
      probability_up: forecast.probability_up,
      probability_down: forecast.probability_down,
      reliability: null,
      signal_state: 'UNKNOWN',
      reliability_reason: 'The reliability service is unavailable, so this direction is unqualified. A directional probability is not a reliability score.',
      model: forecast.model,
      feature_set: forecast.feature_set,
      model_version: forecast.model_version,
      model_timestamp: forecast.issued_at,
      latest_candle: forecast.latest_candle,
      reference_price: forecast.current_price
    }]);
    return {
      degraded: true,
      context: {
        data_fresh: null,
        market: {price: market.price, regime: market.regime || null, timestamp: market.timestamp},
        sentiment: {},
        last_refresh: null,
        forecasts: Object.fromEntries(entries),
        warnings: ['The analyst service is unavailable. Forecasts and monitoring are unaffected; reliability gating cannot be confirmed.']
      },
      configuration: {}
    };
  }

  async function loadContext() {
    const mine = ++generation;
    try {
      const payload = await json('/context');
      if (mine !== generation) return;
      state.context = payload;
      state.contextError = false;
      emit('context', payload);
    } catch {
      try {
        const payload = await loadFallback();
        if (mine !== generation) return;
        state.context = payload;
        state.contextError = false;
        emit('context', payload);
      } catch {
        if (mine !== generation) return;
        state.contextError = true;
        emit('context-error');
      }
    }
  }

  /* The asset list comes from /assets, which reports an asset as deployed only
     when its trained artifacts were actually loaded. Nothing here decides on
     its own which coins are forecastable. */
  function setAsset(code) {
    const next = String(code).toUpperCase();
    const info = state.assets.find(item => item.code === next);
    if (!info || next === state.asset) return;
    state.asset = next;
    document.querySelectorAll('.asset-card').forEach(card => {
      const active = card.dataset.asset === next;
      card.classList.toggle('active', active);
      card.setAttribute('aria-pressed', String(active));
    });
    document.body.dataset.assetDeployed = String(info.deployed);
    document.querySelectorAll('.horizon-control button').forEach(button => { button.disabled = !info.deployed; });
    document.querySelectorAll('[data-asset-field]').forEach(element => {
      element.textContent = info[element.dataset.assetField] || '';
    });
    try { localStorage.setItem('cryptolens-asset', next); } catch { /* private mode */ }
    emit('asset', info);
  }

  const MARKS = {BTC: '₿', ETH: 'Ξ', ADA: '₳'};
  const money = value => value == null ? '—'
    : '$' + Number(value).toLocaleString('en-US',
        {minimumFractionDigits: value < 10 ? 4 : 2, maximumFractionDigits: value < 10 ? 4 : 2});

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  function renderAssetCards() {
    const host = document.getElementById('asset-cards');
    if (!host) return;
    host.replaceChildren(...state.assets.map(item => {
      const card = node('button', 'asset-card');
      card.type = 'button';
      card.dataset.asset = item.code;
      card.dataset.deployed = String(item.deployed);
      card.setAttribute('aria-pressed', String(item.code === state.asset));
      if (item.code === state.asset) card.classList.add('active');

      const head = node('span', 'asset-card-head');
      const mark = node('i', 'asset-mark', MARKS[item.code] || item.code.slice(0, 1));
      mark.dataset.code = item.code;
      head.append(mark, node('span', 'asset-card-name', item.name), node('span', 'asset-card-code', item.code));

      const quote = state.prices[item.code] || {};
      const change = quote.change_24h;
      const changeEl = node('span', 'asset-card-change' + (change == null ? '' : change < 0 ? ' down' : ' up'));
      changeEl.append(node('span', null, change == null ? 'Quote unavailable'
        : `${change < 0 ? '↓' : '↗'} ${change >= 0 ? '+' : ''}${(change * 100).toFixed(2)}%`));
      if (change != null) changeEl.append(node('small', null, '24h'));

      const foot = node('span', 'asset-card-foot');
      if (item.deployed) {
        const label = node('span', 'prob-label');
        label.append(node('span', null, 'UP score'), node('b', 'prob-value', '—'));
        const track = node('span', 'prob-track');
        track.append(node('i', 'prob-fill'));
        foot.append(label, track);
      } else {
        // A quote is available for every coin; a forecast is not. Say so here
        // rather than leaving a gap that reads as missing data.
        foot.append(node('span', 'asset-card-status', 'No model yet'));
      }

      card.append(head, node('strong', 'asset-card-price', money(quote.price)), changeEl, foot);
      card.addEventListener('click', () => setAsset(item.code));
      return card;
    }));
  }

  async function loadPrices() {
    try {
      state.prices = (await json('/prices')).prices || {};
    } catch {
      return;   // keep the last quotes; the cards say so if there were none
    }
    renderAssetCards();
    emit('prices', state.prices);
  }


  /* Offline holdout metrics, as recorded at training time. They are a property
     of the artifact, not of live operation, so one fetch is enough. */
  async function loadHoldout() {
    try { state.holdout = await json('/forecasts'); }
    catch { return; }
    emit('holdout', state.holdout);
  }


  /* The six frozen-model outputs. Deterministic per closed candle, so a short
     poll is enough; the backend caches and will not recompute within an hour. */
  async function loadPredictions() {
    try {
      state.predictions = await json('/predictions');
    } catch (error) {
      emit('predictions-error', 'Predictions are unavailable right now. The models decline to run on incomplete market data rather than guess.');
      return;
    }
    emit('predictions', state.predictions);
  }

  async function loadAssets() {
    try {
      state.assets = (await json('/assets')).assets || [];
    } catch {
      return;   // the dashboard stays on its default single-asset behaviour
    }
    renderAssetCards();
    emit('assets', state.assets);
    loadPrices();
    let stored = null;
    try { stored = localStorage.getItem('cryptolens-asset'); } catch { /* private mode */ }
    const initial = state.assets.find(item => item.code === stored)
      || state.assets.find(item => item.deployed) || state.assets[0];
    if (initial) setAsset(initial.code);
  }

  document.querySelectorAll('[data-horizon]').forEach(button =>
    button.addEventListener('click', () => setHorizon(button.dataset.horizon)));

  document.addEventListener('visibilitychange', () => { if (!document.hidden) loadContext(); });
  window.addEventListener('focus', loadContext);
  loadAssets();
  loadContext();
  loadHoldout();
  loadPredictions();
  setInterval(() => { if (!document.hidden) loadPredictions(); }, 300000);
  setInterval(() => { if (!document.hidden) loadPrices(); }, 60000);
  setInterval(() => { if (!document.hidden) loadContext(); }, 30000);

  return {
    on,
    setHorizon,
    setAsset,
    refreshContext: loadContext,
    get horizon() { return state.horizon; },
    get asset() { return state.asset; },
    get assetInfo() { return current(); },
    get assets() { return state.assets; },
    get deployed() { return Boolean(current()?.deployed); },
    get context() { return state.context; },
    get holdout() { return state.holdout; },
    get predictions() { return state.predictions; }
  };
})();
