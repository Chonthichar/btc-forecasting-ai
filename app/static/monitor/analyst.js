(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const store = window.CryptoLens;
  const questions =['Summarize BTC now', 'Why is BTC moving?', 'Compare 1h / 6h / 24h', 'Explain the 6h forecast', 'What is the current sentiment?', 'Show latest market-moving news', 'Which signal is most reliable?', 'What are the bullish and bearish factors?'];
  const labels = {forecast:'Forecast', research:'Research', review:'Review', validation:'Validation', decision:'Decision'};
  let history = [], busy = false, epoch = 0, periodicBusy = false, modelName = 'OpenAI';
  const welcome = document.querySelector('.chat-welcome').cloneNode(true);
  const percent = value => value == null ? 'Unavailable' : `${(value * 100).toFixed(1)}%`;
  function utc(value) {
    if (!value || !Number.isFinite(Date.parse(value))) return 'Unavailable';
    return new Date(value).toISOString().slice(0, 16).replace('T', ' ') + ' UTC';
  }
  function node(tag, text, className) {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (className) element.className = className;
    return element;
  }
  function link(url, text, className) {
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) return node('span', text);
      const element = node('a', text, className);
      element.href = parsed.href; element.target = '_blank'; element.rel = 'noopener noreferrer';
      return element;
    } catch { return node('span', text); }
  }
  async function api(path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), body ? 240000 : 15000);
    try {
      const response = await fetch(path, {method:body ? 'POST' : 'GET', cache:'no-store', signal:controller.signal,
        ...(body ? {headers:{'Content-Type':'application/json'},body:JSON.stringify(body)} : {})});
      if (!response.ok) throw new Error(response.status === 503 ? 'The analyst is temporarily unavailable. Please retry shortly.' : 'The request could not be completed. Please retry.');
      return await response.json();
    } finally { clearTimeout(timer); }
  }
  const horizonLabel = horizon => horizon === '1' ? '1 hour' : `${horizon} hours`;



  function renderContext(payload) {
    const context = payload.context, configuration = payload.configuration;
    $('market-live').textContent = context.data_fresh === true ? '● LIVE' : '○ OFFLINE';
    $('market-live').className = 'live-state' + (context.data_fresh === true ? ' live' : '');
    $('market-live').title = 'LIVE means the saved hourly market snapshot is fresh; this is not a streaming price feed.';
    $('ai-price').textContent = context.market.price == null ? 'Unavailable' : '$' + context.market.price.toLocaleString('en-US', {minimumFractionDigits:2,maximumFractionDigits:2});
    $('ai-price').title = 'Source candle opened: ' + utc(context.market.timestamp);
    $('ai-regime').textContent = context.market.regime || 'Not provided';
    $('ai-sentiment').textContent = context.sentiment.label || (context.sentiment.score == null ? 'Not provided' : `Score ${context.sentiment.score.toFixed(3)}`);
    $('ai-refreshed').textContent = utc(context.last_refresh);
    if (payload.degraded) {
      $('reliability-threshold').textContent = 'RELIABILITY GATE · UNAVAILABLE';
      $('provider-status').textContent = 'The analyst service is unavailable. Saved forecasts and monitoring are unaffected.';
    } else {
      // A threshold implies a gate that can fire. With no reliability model
      // deployed every horizon reports null, so say that instead.
      const gated = Object.values(context.forecasts || {}).some(entry => entry?.reliability != null);
      $('reliability-threshold').textContent = gated
        ? `RELIABILITY GATE · ${percent(configuration.reliability_threshold)}`
        : 'RELIABILITY GATE · NOT DEPLOYED';
      $('reliability-threshold').title = gated
        ? 'A forecast is only issued as a signal when its validated reliability clears this threshold.'
        : 'No model produces a reliability score yet, so no forecast can be qualified as a signal.';
      modelName = configuration.openai_model || 'OpenAI';
      $('provider-status').textContent = `${configuration.agent_framework || "OpenAI"} | ${configuration.openai_configured ? modelName + ' · configured' : 'not configured'} · Tavily ${configuration.tavily_configured ? 'configured' : 'not configured'} · News cache ${configuration.news_cache_minutes} min`;
      $('provider-status').title = 'Keys are read by the API from the local .env file. Configured means present, not yet verified by a provider request.';
    }
  }

  function contextUnavailable() {
    $('market-live').textContent = '○ OFFLINE';
    $('market-live').className = 'live-state';
  }
  function renderEvidence(research) {
    const items = research.evidence || [];
    $('evidence-status').textContent = `${items.length} sources · ${research.status.replace('_',' ')}${research.cached ? ' · cached' : ''}`;
    $('evidence-footer').textContent = research.retrieved_at ? `Retrieved ${utc(research.retrieved_at)}. News never increases model reliability.` : 'News never increases model reliability.';
    if (!items.length) {
      const empty = node('div', null, 'evidence-empty');
      empty.append(node('span','◎'),node('strong',research.status === 'unavailable' ? 'News analysis unavailable' : 'No news analysis yet.'),
        node('p',research.status === 'not_requested' ? 'Ask a current-news question or refresh this section. The 30-second dashboard update does not call research providers.' : research.reason || 'No sufficiently relevant sources passed validation.'));
      $('evidence-list').replaceChildren(empty); return;
    }
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      const card = node('article', null, 'evidence-item');
      const top = node('div', null, 'evidence-top');
      const icon = node('span', item.category.slice(0, 1), 'evidence-icon');
      const meta = node('div', `${item.source} · ${item.category.replace('_',' ')}`, 'evidence-meta');
      const stance = node('span', item.direction[0].toUpperCase() + item.direction.slice(1), 'evidence-tag ' + item.direction);
      top.append(icon, meta, stance);
      const heading = node('h3'); heading.append(link(item.url,item.headline));
      card.append(top,heading,node('p',item.summary),node('div',`Published ${utc(item.published_at)}${item.recency_verified ? '' : ' · time unverified'}`,'evidence-time'),link(item.url,'Read analysis →','source-link'));
      fragment.append(card);
    }
    $('evidence-list').replaceChildren(fragment);
  }
  function activity(states = {}) {
    for (const [key,label] of Object.entries(labels)) {
      const state = states[key] || 'idle';
      const node = $('agent-' + key);
      node.dataset.state = state;
      node.querySelector('.agent-state').textContent = state;
      node.setAttribute('aria-label', `${label}: ${state}`);
      // The connector feeding a node carries that node's state, so the line
      // animates while the stage it leads into is working.
      const link = node.previousElementSibling;
      if (link && link.classList.contains('agent-link')) link.dataset.state = state;
    }
    if (states.research === 'running') $('chat-status').textContent = 'Tavily is searching current market sources…';
    else if (states.review === 'running') $('chat-status').textContent = 'Reviewing retrieved sources independently…';
    else if (states.validation === 'running') $('chat-status').textContent = 'Checking forecast values and source quality…';
    else if (states.decision === 'running') $('chat-status').textContent = 'Preparing the answer from validated facts…';
  }
  function message(role, content, sources = [], status = '') {
    if ($('chat-log').querySelector('.chat-welcome')) $('chat-log').replaceChildren();
    const entry = node('article', null, 'chat-message ' + role);
    const label = node('div', null, 'message-label');
    label.append(node('span',role === 'user' ? 'YOU' : 'AI MARKET ANALYST'),node('span',status));
    entry.append(label,node('div',content,'message-content'));
    if (sources.length) {
      const links = node('div',null,'message-sources');
      sources.forEach((source,index) => links.append(link(source.url,`${index + 1}. ${source.source} ↗`,'source-link')));
      entry.append(links);
    }
    $('chat-log').append(entry); $('chat-log').scrollTop = $('chat-log').scrollHeight;
  }
  function setBusy(value) {
    busy = value; $('send-chat').disabled = value; $('research-news').disabled = value;
    $('quick-questions').querySelectorAll('button').forEach(button => { button.disabled = value; });
    $('chat-form').setAttribute('aria-busy', String(value));
  }
  async function run(kind, question) {
    if (busy) return;
    setBusy(true);
    const generation = epoch, requestId = crypto.randomUUID();
    const previous = history.slice(-12).map(item => ({role:item.role,content:item.content.slice(0,6000)}));
    if (kind === 'chat') { message('user',question); $('chat-input').value = ''; }
    $('chat-status').textContent = kind === 'chat' ? 'Reading saved model context…' : 'Tavily is researching current sources…';
    activity({forecast:'running'});
    let polling = false;
    const timer = setInterval(async () => {
      if (polling || generation !== epoch) return;
      polling = true;
      try { const data = await api('/agents/status?request_id=' + encodeURIComponent(requestId)); if (generation === epoch) activity(data.agent_status); }
      catch { /* The main request reports connection failures. */ }
      finally { polling = false; }
    }, 1100);
    try {
      const data = await api('/' + kind, kind === 'chat' ? {message:question,history:previous,request_id:requestId} : {question,request_id:requestId,fresh_research:true});
      if (generation !== epoch) return;
      activity(data.agent_status);
      if (kind === 'chat') {
        const answerLabel = data.llm_status === 'ok' ? `OPENAI · ${modelName}` :
          data.llm_status === 'invalid_output' ? 'VERIFIED FALLBACK' : 'FALLBACK · OPENAI UNAVAILABLE';
        message('assistant',data.answer,data.sources,answerLabel);
        history.push({role:'user',content:question},{role:'assistant',content:data.answer}); history = history.slice(-12);
        $('chat-status').textContent = `${data.llm_status === 'ok' ? 'OpenAI answered' : data.llm_status === 'invalid_output' ? 'Verified fallback shown' : 'OpenAI unavailable · fallback shown'} · ${data.research_used ? 'Tavily ' + data.research_status : 'no web search needed'}`;
      } else { renderEvidence(data.research); $('chat-status').textContent = `Research ${data.research.status} · sources validated`; }
      await refresh();
      store.refreshContext();
    } catch (error) {
      if (generation !== epoch) return;
      const explanation = error.name === 'AbortError' ? 'The request timed out. Please try again shortly.' : 'The analyst request could not be completed. Check API health and retry.';
      $('chat-status').textContent = explanation;
      if (kind === 'chat') message('assistant',explanation);
      activity({decision:'unavailable'});
    } finally { clearInterval(timer); setBusy(false); }
  }
  async function refresh() {
    if (periodicBusy) return;
    periodicBusy = true;
    try { renderEvidence(await api('/evidence')); }
    catch { $('evidence-status').textContent = 'Evidence service unavailable · previous sources may be stale'; }
    finally { periodicBusy = false; }
  }
  for (const question of questions) {
    const button = node('button',question); button.type = 'button'; button.addEventListener('click',() => run('chat',question)); $('quick-questions').append(button);
  }
  $('chat-form').addEventListener('submit',event => { event.preventDefault(); const question = $('chat-input').value.trim(); if (question) run('chat',question); });
  $('chat-input').addEventListener('keydown',event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('chat-form').requestSubmit(); } });
  $('research-news').addEventListener('click',() => run('research','Bitcoin latest market news today'));
  $('clear-chat').addEventListener('click',() => { epoch++; history = []; $('chat-log').replaceChildren(welcome.cloneNode(true)); $('chat-status').textContent = busy ? 'Chat cleared · finishing the current request' : 'Ready · research runs only when needed'; activity(); });
  store.on('context', payload => renderContext(payload));
  store.on('context-error', contextUnavailable);
  if (store.context) renderContext(store.context);
  window.addEventListener('focus',refresh);
  refresh(); setInterval(refresh,30000);
})();
