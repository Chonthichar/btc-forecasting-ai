(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const stamp = value => value && Number.isFinite(Date.parse(value)) ? new Date(value).toISOString().slice(0,16).replace('T',' ') + ' UTC' : 'Not yet available';
  let busy = false;
  function text(id, value) { $(id).textContent = value; }
  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      const response = await fetch('/monitoring/activity', {cache:'no-store', signal:AbortSignal.timeout(15000)});
      if (!response.ok) throw new Error('Monitoring API unavailable');
      const data = await response.json(), event = data.last_research, operation = data.research || {};
      text('scan-worker', data.worker_alive ? 'ACTIVE' : 'STOPPED');
      text('scan-last', stamp(data.last_market_scan));
      text('scan-next', stamp(data.next_cycle_due));
      text('scan-success', stamp(data.last_successful_prediction_cycle));
      text('scan-backup', (data.backup?.status || 'UNKNOWN') + (data.backup?.saved_at ? ' · ' + stamp(data.backup.saved_at) : ''));
      text('scan-cadence', `Every ${data.monitor_interval_minutes} minutes; runs while this page is closed.`);
      text('research-state', operation.status || 'IDLE');
      text('research-last-time', stamp(event?.result?.retrieved_at || event?.completed_at));
      const trigger = data.last_research_trigger;
      let reason = trigger?.triggered_by?.map(value => value.replaceAll('_',' ')).join(', ') || 'No meaningful change detected';
      if (trigger?.details?.price_change_1h != null) reason += ` (${trigger.details.price_change_1h.toFixed(2)}% in 1h)`;
      text('research-trigger-reason', reason);
      text('research-event-note', operation.cache_reason || (event?.status === 'error' ? 'Last context research was unavailable. Saved market forecasts are preserved.' : 'Web research is event-triggered to reduce unnecessary API calls.'));
      text('stored-market-context', event?.summary || 'A contextual summary appears after an event produces useful validated evidence.');
      const agents = [['Market monitor','monitor'],['Forecast agent','forecast'],['Trigger engine','trigger'],['Research agent',null],['Validation agent','validation'],['Decision agent','decision']];
      const fragment = document.createDocumentFragment();
      for (const [label,key] of agents) {
        const node = document.createElement('span');
        node.textContent = label + ': ' + (key ? operation[key] || 'idle' : operation.status || 'IDLE');
        fragment.append(node);
      }
      $('continuous-agent-status').replaceChildren(fragment);
      text('scan-error', [data.last_error, data.backup?.error].filter(Boolean).join(' '));
    } catch {
      text('scan-worker','UNAVAILABLE');
      text('research-state','UNAVAILABLE');
      text('scan-error','Cannot confirm current monitoring health; displayed timestamps may be stale.');
    } finally { busy = false; }
  }
  window.addEventListener('focus', refresh);
  refresh(); setInterval(refresh, 30000);
})();
