/* Views only read durable API records. No example forecasts enter this UI. */
'use strict';
const $ = id => document.getElementById(id);
const state = {horizon:1, days:7, version:'', status:'', offset:0, limit:20, series:[], loading:false, generation:0, active:false};
const percent = n => n == null ? '—' : (n*100).toFixed(1)+'%';
const money = n => n == null ? '—' : '$'+Number(n).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
const when = value => value ? new Intl.DateTimeFormat('en-GB',{timeZone:'UTC',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(value)) : '—';
const modelName = value => ({vae_transformer_v2:'VAE-Transformer',gru:'GRU',lstm:'LSTM'}[value] || value || 'Model');
const featureName = value => ({market_only:'Market only',market_plus_sentiment:'Market + sentiment',sentiment_only:'Sentiment only'}[value] || value || '—');
const statusName = row => row.status === 'scored' ? (row.correct ? 'Correct' : 'Incorrect') : ({pending:'Pending',awaiting_market_data:'Awaiting price',excluded_late:'Excluded · late'}[row.status] || row.status);
const setText = (id,value) => { $(id).textContent = value; };
function element(tag,text,className){const node=document.createElement(tag); if(text!=null)node.textContent=text;if(className)node.className=className;return node;}
function notify(message,error=false){$('notice').hidden=!message;$('notice').textContent=message||'';$('notice').classList.toggle('error',error);}
function query(history=false){const p=new URLSearchParams({horizon:state.horizon,days:state.days});if(state.version)p.set('model_version',state.version);if(history){p.set('limit',state.limit);p.set('offset',state.offset);if(state.status)p.set('status',state.status);}return p;}
async function readJson(url){const r=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(20000)});if(!r.ok)throw new Error('API returned '+r.status);return r.json();}

function renderSummary(data){
  const q=data.qualification;
  if(q){
    setText('qualification-performance','Signal coverage: '+percent(q.signal_coverage)+' | NO_SIGNAL: '+percent(q.no_signal_frequency)+' | UNKNOWN: '+percent(q.unknown_frequency)+' | Qualified accuracy: '+percent(q.qualified_performance.accuracy));
    const groups=$('qualification-groups');groups.replaceChildren();
    for(const [label,values] of [['Reliability',q.reliability_buckets],['Regime',q.regimes]]){
      for(const [key,m] of Object.entries(values)){groups.append(element('span',label+' '+key+': '+percent(m.accuracy)+' accuracy / '+m.scored+' scored'));}
    }
    if(q.legacy_unannotated)groups.append(element('span',q.legacy_unannotated+' older forecasts have no saved issue-time gate; excluded from coverage.'));
  }

  const sys=data.system, model=data.model, latest=data.latest;
  setText('crumb','btc-direction-'+state.horizon+'h');
  setText('model-name',modelName(latest?.model || model.model));
  setText('feature-set',featureName(latest?.feature_set || model.feature_set));
  setText('horizon-description',state.horizon+'h horizon');
  setText('accuracy',percent(data.accuracy));setText('scored',data.scored.toLocaleString());
  setText('pending',data.pending.toLocaleString());setText('brier',data.brier_score==null?'—':data.brier_score.toFixed(3));
  setText('accuracy-detail',data.scored ? data.correct+' correct / '+data.scored+' scored'+(data.scored<30?' · small sample':'') : 'Awaiting outcomes');
  const delta=data.accuracy==null?null:(data.accuracy-data.baseline_accuracy)*100;
  setText('baseline-detail',delta==null?'Compared with always predicting UP':(delta>=0?'+':'')+delta.toFixed(1)+' pp vs always UP ('+percent(data.baseline_accuracy)+')');
  setText('scored-detail',data.scored?state.days+'-day window · this model version':'No outcomes scored yet');
  setText('pending-detail',data.pending?'Outcome due after '+state.horizon+'h target closes':data.total?'No future outcomes pending':'Waiting for the first forecast');
  setText('missing-detail',data.awaiting_market_data?data.awaiting_market_data+' overdue · waiting for market data':data.excluded?data.excluded+' issued too late · excluded':'Predictions are saved before evaluation');
  state.series=data.series||[]; drawChart();
  setText('chart-description','Trailing '+state.days+'-day accuracy · scored forecasts only');
  setText('window-note','Window by issue time · selected model version · '+data.scored+' scored');
  const oldVersion=data.model_version!==model.model_version;
  setText('model-badge',oldVersion?'ARCHIVED VERSION':sys.last_error?'NEEDS ATTENTION':data.scored?'MONITORING':'COLLECTING');
  $('model-badge').classList.toggle('error',Boolean(sys.last_error));
  $('latest-empty').hidden=Boolean(latest);$('latest-content').hidden=!latest;
  if(latest){
    setText('direction',latest.predicted_up?'↗ UP':'↘ DOWN');$('direction').classList.toggle('down',!latest.predicted_up);
    setText('probability',percent(latest.probability_up));$('probability-fill').style.width=(latest.probability_up*100)+'%';
    setText('reference-price',money(latest.current_price));setText('issued-at',when(latest.issued_at)+' UTC');
    setText('target-at',when(latest.target_close)+' UTC');setText('latest-status',statusName(latest));
  }
  const select=$('version-select');select.replaceChildren();
  for(const v of data.versions){const current=v.model_version===model.model_version;const option=element('option',(current?'Current · ':'')+v.model_version.slice(0,8));option.value=current?'':v.model_version;select.append(option);}
  select.value=state.version;
  state.active=Boolean(sys.active_run?.id);
  $('refresh-button').disabled=state.active;
  $('refresh-button').querySelector('span').textContent=state.active?'Refresh running…':'Run refresh';
  setText('worker-badge',state.active?'REFRESHING':sys.worker_alive?'WORKER ONLINE':'WORKER OFFLINE');
  $('worker-badge').classList.toggle('error',!sys.worker_alive&&!state.active);
  setText('last-success',sys.last_success?when(sys.last_success)+' UTC':'Not yet completed');
  setText('candle-freshness',sys.latest_candle_close?when(sys.latest_candle_close)+(sys.data_stale?' · STALE':' UTC'):'Awaiting market data');
  const lastRun=(sys.runs||[]).find(r=>r.duration_ms!=null);
  setText('run-duration',lastRun?(lastRun.duration_ms/1000).toFixed(1)+' seconds':'—');
  const newsHours=sys.quality?.hours_with_news_in_latest_sequence;
  setText('sentiment-coverage',newsHours==null?'Awaiting news collection':newsHours+' of 48 hours with news');
  const issues=[];
  if(sys.last_error)issues.push('Last refresh failed: '+sys.last_error);
  if(sys.last_evaluation_error)issues.push('Outcome evaluation: '+sys.last_evaluation_error);
  if(sys.data_stale)issues.push('Market snapshot is stale. Previous forecasts remain in the history.');
  if(state.active)issues.push('Refresh in progress. Market and news collection can take several minutes.');
  else if(sys.worker_alive&&sys.worker?.next_run_at)issues.push('Next scheduled cycle: '+when(sys.worker.next_run_at)+' UTC.');
  else if(!sys.worker_alive)issues.push('Hourly worker is not reporting. Run a manual refresh or start the monitoring container.');
  setText('health-message',issues.join(' ')||'Monitoring is running. New outcomes are evaluated each hour.');
  const runs=$('recent-runs');runs.replaceChildren();
  for(const run of (sys.runs||[]).slice(0,3)){const row=element('div',null,'run-row');row.append(element('span',when(run.started_at)+' UTC'),element('span',run.trigger),element('span',run.status,run.status==='error'||run.status==='failed'?'run-error':run.status==='success'?'run-success':''));runs.append(row);}
  if(sys.last_error)notify('The last collection did not finish. Your saved history is intact; the worker will retry. See System health below.',true);
  else if(!state.active&&!sys.worker_alive)notify('The hourly worker is offline. This page can still show saved predictions.',true);
  else notify('');
  setText('connection-label','API connected');$('connection-dot').classList.remove('error');
  setText('last-updated','Updated '+when(data.generated_at)+' UTC');
}

function renderHistory(data){
  const body=$('history-rows');body.replaceChildren();
  if(!data.items.length){const row=element('tr'),cell=element('td');cell.colSpan=7;const content=element('div',state.status?'No forecasts match this filter.':'No recorded forecasts in this window.','table-empty');content.append(element('small','Only forecasts saved during live operation appear here.'));cell.append(content);row.append(cell);body.append(row);}
  for(const item of data.items){const row=element('tr');
    [when(item.issued_at),item.predicted_up?'↗ UP':'↘ DOWN',percent(item.probability_up),money(item.current_price),money(item.target_price),when(item.target_close)].forEach(value=>row.append(element('td',value)));
    const cell=element('td'),tag=element('span',statusName(item),'result-tag '+(item.status==='scored'?(item.correct?'correct':'incorrect'):'pending'));cell.append(tag);row.append(cell);body.append(row);
  }
  const first=data.total?state.offset+1:0,last=Math.min(state.offset+state.limit,data.total);
  setText('history-count',first+'–'+last+' of '+data.total+' forecasts');
  $('previous-page').disabled=state.offset===0;$('next-page').disabled=last>=data.total;
}

function drawChart(){
  const canvas=$('accuracy-chart'),box=canvas.getBoundingClientRect(),ratio=window.devicePixelRatio||1;
  if(!box.width)return;
  canvas.width=box.width*ratio;canvas.height=box.height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.font='9px monospace';
  const left=43,right=box.width-15,top=20,bottom=box.height-29;
  for(let i=0;i<5;i++){const v=1-i/4,y=top+(bottom-top)*i/4;ctx.fillStyle='#607c8c';ctx.fillText(Math.round(v*100)+'%',4,y+3);ctx.beginPath();ctx.setLineDash([3,5]);ctx.strokeStyle='#293a44';ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();}
  const series=state.series,n=series.length,x=i=>left+(right-left)*(n<=1?.5:i/(n-1)),y=v=>bottom-v*(bottom-top);
  let valid=0;
  for(const [key,color,dash] of [['baseline_accuracy','#8586b4',[4,4]],['accuracy','#28c4d4',[]]]){
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=key==='accuracy'?2:1.3;ctx.setLineDash(dash);ctx.beginPath();let drawing=false;
    series.forEach((p,i)=>{if(p[key]==null){drawing=false;return;}if(key==='accuracy')valid++;if(!drawing)ctx.moveTo(x(i),y(p[key]));else ctx.lineTo(x(i),y(p[key]));drawing=true;});ctx.stroke();ctx.setLineDash([]);
    series.forEach((p,i)=>{if(p[key]!=null){ctx.beginPath();ctx.arc(x(i),y(p[key]),key==='accuracy'?2.7:1.8,0,Math.PI*2);ctx.fill();}});
  }
  ctx.fillStyle='#607c8c';ctx.textAlign='center';
  const tickIndices=[...new Set([0,Math.round((n-1)/3),Math.round((n-1)*2/3),n-1])];
  for(const i of tickIndices){if(series[i])ctx.fillText(new Intl.DateTimeFormat('en-GB',{timeZone:'UTC',month:'short',day:'numeric'}).format(new Date(series[i].date+'T00:00:00Z')),x(i),box.height-7);}
  $('chart-empty').hidden=valid>0;
  canvas.setAttribute('aria-label',valid?'Rolling '+state.days+'-day accuracy and always UP baseline; current accuracy '+percent(series[n-1]?.accuracy):'No scored forecasts yet. Accuracy appears after target candles close.');
}

async function load(){
  const generation=++state.generation;
  try{
    const [summary,history]=await Promise.all([readJson('/monitoring/summary?'+query()),readJson('/monitoring/history?'+query(true))]);
    if(generation!==state.generation)return;
    renderSummary(summary);renderHistory(history);
  }catch(error){if(generation!==state.generation)return;notify('Unable to update from the API. Displayed values may be stale. Retrying automatically.',true);setText('connection-label','API unavailable');$('connection-dot').classList.add('error');setText('last-updated','Connection lost · displayed data may be stale');}
}
document.querySelectorAll('[data-horizon]').forEach(button=>button.addEventListener('click',()=>{
  state.horizon=Number(button.dataset.horizon);state.version='';state.offset=0;
  document.querySelectorAll('[data-horizon]').forEach(b=>{const active=b===button;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});load();
}));
document.querySelectorAll('[data-days]').forEach(button=>button.addEventListener('click',()=>{
  state.days=Number(button.dataset.days);state.offset=0;
  document.querySelectorAll('[data-days]').forEach(b=>{const active=b===button;b.classList.toggle('selected',active);b.setAttribute('aria-pressed',active);});load();
}));
$('version-select').addEventListener('change',event=>{state.version=event.target.value;state.offset=0;load();});
$('status-filter').addEventListener('change',event=>{state.status=event.target.value;state.offset=0;load();});
$('previous-page').addEventListener('click',()=>{state.offset=Math.max(0,state.offset-state.limit);load();});
$('next-page').addEventListener('click',()=>{state.offset+=state.limit;load();});
$('refresh-button').addEventListener('click',async()=>{
  $('refresh-button').disabled=true;
  try{const result=await fetch('/monitoring/run?background=true',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trigger:'manual'}),signal:AbortSignal.timeout(15000)});if(!result.ok)throw new Error('Refresh failed');state.active=true;notify('Refresh requested. New predictions will appear when collection finishes.');setTimeout(load,800);}
  catch(error){notify('Unable to request a refresh. Check the API connection and try again.',true);$('refresh-button').disabled=false;}
});
document.querySelectorAll('.nav-item').forEach(link=>link.addEventListener('click',()=>{document.querySelectorAll('.nav-item').forEach(a=>a.classList.toggle('active',a===link));}));
window.addEventListener('resize',drawChart);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});
drawChart();load();
let ticks=0;setInterval(()=>{ticks++;if(!document.hidden&&(state.active||ticks%6===0))load();},5000);
