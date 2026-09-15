(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const stamp = value => value && Number.isFinite(Date.parse(value)) ? new Date(value).toISOString().slice(0,16).replace('T',' ') + ' UTC' : 'Not yet available';
  let busy = false;
  function text(id, value) { $(id).textContent = value; }
  function tile(id, value, detail, state) {
    const node = $(id);
    if (!node) return;
    node.dataset.state = state || 'idle';
    node.querySelector('.hc-value').textContent = value;
    node.querySelector('.hc-detail').textContent = detail;
  }
  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      const response = await fetch('/monitoring/activity', {cache:'no-store', signal:AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error('Monitoring API unavailable');
      const data = await response.json(), event = data.last_research;
      text('scan-worker', data.worker_alive ? 'ACTIVE' : 'STOPPED');
      text('scan-last', stamp(data.last_market_scan));
      text('scan-next', stamp(data.next_cycle_due));
      text('scan-success', stamp(data.last_successful_prediction_cycle));
      text('scan-backup', (data.backup?.status || 'UNKNOWN') + (data.backup?.saved_at ? ' · ' + stamp(data.backup.saved_at) : ''));
      const backup = (data.backup?.status || '').toUpperCase();
      const backupDetail = data.backup?.error ? data.backup.error
        : data.backup?.saved_at ? stamp(data.backup.saved_at)
        : backup === 'LOCAL_ONLY' ? 'No backup repository configured'
        : 'Not yet saved';
      tile('hc-backup', backup || '—', backupDetail,
        !backup ? 'idle' : data.backup?.error ? 'bad'
          : ['OK','RESTORED','CONFIGURED','LOCAL_ONLY'].includes(backup) ? 'ok' : 'warn');
      text('scan-cadence', `Every ${data.monitor_interval_minutes} minutes; runs while this page is closed.`);
      text('research-last-time', stamp(event?.result?.retrieved_at || event?.completed_at));
      const trigger = data.last_research_trigger;
      let reason = trigger?.triggered_by?.map(value => value.replaceAll('_',' ')).join(', ') || 'No meaningful change detected';
      if (trigger?.details?.price_change_1h != null) reason += ` (${trigger.details.price_change_1h.toFixed(2)}% in 1h)`;
      text('research-trigger-reason', reason);
      text('scan-error', [data.last_error, data.backup?.error].filter(Boolean).join(' '));
    } catch {
      text('scan-worker','UNAVAILABLE');
      tile('hc-backup','—','Monitoring API unreachable','warn');
      text('scan-error','Cannot confirm current monitoring health; displayed timestamps may be stale.');
    } finally { busy = false; }
  }
  const modelName = value => ({vae_transformer_v2:'VAE-Transformer', gru:'GRU', lstm:'LSTM'}[value] || value || 'Model');
  const featureName = value => ({market_only:'Market only', market_plus_sentiment:'Market + sentiment', sentiment_only:'Sentiment only'}[value] || value || '—');

  async function loadDeployedModels() {
    const target = $('deployed-models');
    try {
      const response = await fetch('/system', {cache:'no-store', signal:AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error('API returned ' + response.status);
      const models = (await response.json()).deployed_models || {};
      const cards = Object.entries(models).sort((a, b) => Number(a[0]) - Number(b[0])).map(([horizon, spec]) => {
        const card = document.createElement('article');
        card.className = 'deployed-card';
        const head = document.createElement('span');
        head.className = 'deployed-horizon';
        head.textContent = horizon === '1' ? '1 hour' : horizon + ' hours';
        const name = document.createElement('strong');
        name.textContent = modelName(spec.model);
        const features = document.createElement('span');
        features.className = 'deployed-features';
        features.textContent = featureName(spec.feature_set);
        const version = document.createElement('code');
        version.textContent = (spec.model_version || '').slice(0, 8);
        version.title = 'Fingerprint of the model weights, scalers, config and feature code: ' + spec.model_version;
        card.append(head, name, features, version);
        return card;
      });
      target.replaceChildren(...(cards.length ? cards : [Object.assign(document.createElement('p'), {className:'signal-note', textContent:'No deployment registry entries found.'})]));
    } catch {
      target.replaceChildren(Object.assign(document.createElement('p'), {className:'signal-note', textContent:'Deployment registry unavailable.'}));
    }
  }

  window.addEventListener('focus', refresh);
  refresh(); setInterval(refresh, 30000);
  loadDeployedModels();
})();
