/* Renders the six frozen-model outputs from /predictions.

   These numbers are displayed exactly as the backend returns them. Nothing here
   rounds toward a friendlier value, blends horizons, or fills a gap with a
   guess: if the API declines to predict, the panels say so. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const store = window.CryptoLens;
  const percent = value => value == null ? '—' : (Number(value) * 100).toFixed(2) + '%';
  const utc = value => {
    const at = Date.parse(value);
    return Number.isFinite(at) ? new Date(at).toISOString().slice(0, 16).replace('T', ' ') + ' UTC' : '—';
  };

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text != null) element.textContent = text;
    return element;
  }

  /* The asset card carries the selected horizon's direction. It reads the same
     /predictions payload as the panel below it: two sources on one screen
     disagreed on magnitude and, at 6h, on the direction itself. */
  function updateAssetCard(payload) {
    const card = document.querySelector('.asset-card[data-deployed="true"]');
    if (!card) return;
    const entry = (payload.direction || {})[String(store.horizon)] || {};
    const label = entry.label;
    card.classList.toggle('qualified-up', label === 'UP');
    card.classList.toggle('qualified-down', label === 'DOWN');
    const value = card.querySelector('.prob-value');
    const fill = card.querySelector('.prob-fill');
    if (value) value.textContent = percent(entry.up_score);
    if (fill) fill.style.width = entry.up_score == null ? '0' : `${entry.up_score * 100}%`;
  }

  function renderDirection(direction) {
    const host = $('horizon-bars');
    if (!host) return;
    host.replaceChildren(...['1', '6', '24'].map(key => {
      const entry = direction[key] || {};
      const up = entry.up_score;
      const label = entry.label || '—';
      const card = node('div', 'horizon-bar' + (label === 'UP' ? ' qualified-up' : label === 'DOWN' ? ' qualified-down' : ''));
      card.setAttribute('role', 'listitem');
      card.setAttribute('aria-label',
        `${key} hour direction ${label}, up score ${percent(up)}, down score ${percent(entry.down_score)}`);
      const track = node('div', 'bar-track');
      const fill = node('div', 'bar-fill');
      fill.style.height = up == null ? '0' : `${Math.max(up * 100, 1.5)}%`;
      track.append(node('i', 'bar-midline'), fill);
      card.append(
        node('strong', 'bar-number', percent(up)),
        track,
        node('span', 'bar-label', key === '1' ? '1 hour' : `${key} hours`),
        node('span', 'bar-state', label),
        node('span', 'bar-down', `DOWN ${percent(entry.down_score)}`));
      return card;
    }));
  }

  function renderRisk(risk) {
    const host = $('risk-cards');
    if (!host) return;
    host.replaceChildren(...['1', '6', '24'].map(key => {
      const entry = risk[key] || {};
      const level = entry.risk_level || 'UNKNOWN';
      const card = node('div', 'risk-card');
      card.dataset.level = level;
      card.setAttribute('role', 'listitem');
      card.setAttribute('aria-label',
        `${key} hour large movement risk ${percent(entry.model_score)}, band ${level}, ` +
        `historical event rate ${percent(entry.historical_event_rate)}`);

      const head = node('div', 'risk-head');
      head.append(node('span', 'risk-horizon', key === '1' ? '1 hour' : `${key} hours`),
                  node('span', 'risk-band', level));
      const facts = node('dl', 'risk-facts');
      for (const [term, value] of [
        ['Historical rate for this band', percent(entry.historical_event_rate)],
        ['Base rate, all hours', percent(entry.baseline_event_rate)]
      ]) {
        const row = node('div');
        row.append(node('dt', null, term), node('dd', null, value));
        facts.append(row);
      }
      card.append(head, node('strong', 'risk-score', percent(entry.model_score)), facts);
      return card;
    }));
  }

  function unavailable(message) {
    for (const [id, text] of [['horizon-bars', message], ['risk-cards', message]]) {
      const host = $(id);
      if (host) host.replaceChildren(node('p', 'signal-note', text));
    }
  }

  function apply(payload) {
    renderDirection(payload.direction || {});
    updateAssetCard(payload);
    renderRisk(payload.movement_risk || {});
    const stamp = $('direction-candle');
    if (stamp) stamp.textContent = 'CANDLE ' + utc(payload.forecast_candle_utc);
    const event = $('risk-event');
    if (event && payload.movement_risk?.['24']?.event_definition) {
      event.textContent = payload.movement_risk['24'].event_definition
        .replace(' at any point within the next 24 hours', ', within each horizon');
    }
  }

  store.on('predictions', apply);
  // the card follows the horizon selector, and the cards are rebuilt on quote refresh
  for (const event of ['horizon', 'prices', 'assets', 'asset']) {
    store.on(event, () => { if (store.predictions) updateAssetCard(store.predictions); });
  }
  store.on('predictions-error', detail => unavailable(
    detail || 'Predictions are unavailable. The models refuse to run on incomplete market data rather than guess.'));
  if (store.predictions) apply(store.predictions);
})();
