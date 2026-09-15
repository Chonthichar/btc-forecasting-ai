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

function renderQualification(q){
  const body=$('qualification-groups'), summary=$('qualification-summary');
  if(!q){setText('qualification-performance','Awaiting issue-time signal records.');return;}
  setText('qualification-performance','Accuracy by the reliability band assigned when each forecast was issued');
  summary.replaceChildren();
  for(const [label,value,hint] of [
    ['Qualified accuracy',percent(q.qualified_performance?.accuracy),(q.qualified_performance?.scored||0)+' scored'],
    ['Signal coverage',percent(q.signal_coverage),'forecasts that passed the gate'],
    ['NO_SIGNAL',percent(q.no_signal_frequency),'gate withheld a direction'],
    ['UNKNOWN',percent(q.unknown_frequency),'inputs incomplete or stale']
  ]){
    const tile=element('div',null,'qual-tile');
    tile.append(element('span',label),element('strong',value),element('small',hint));
    summary.append(tile);
  }
  const rows=[];
  for(const [label,values] of [['Reliability',q.reliability_buckets],['Regime',q.regimes]]){
    for(const [key,m] of Object.entries(values||{})){
      const row=element('tr');
      row.append(element('td',label),element('td',key),element('td',percent(m.accuracy)),element('td',String(m.scored)));
      rows.push(row);
    }
  }
  if(rows.length)body.replaceChildren(...rows);
  else{
    const row=element('tr'),cell=element('td');cell.colSpan=4;
    cell.append(element('div','Awaiting scored forecasts with saved issue-time annotations.','table-empty'));
    row.append(cell);body.replaceChildren(row);
  }
  // coverage 0 with everything UNKNOWN is not a data gap: no reliability model
  // is deployed, so the gate has nothing to act on.
  // coverage arrives as 0 live and null before any annotated forecast exists;
  // either way nothing has been qualified, and the cause is the same.
  const ungated=!q.signal_coverage&&!(q.qualified_performance&&q.qualified_performance.scored);
  if(ungated)setText('qualification-performance',
    'No reliability model is deployed, so no forecast can be qualified and coverage is 0%. '+
    'Qualifying a signal needs a second-stage model that estimates whether the primary call is correct, '+
    'trained on out-of-fold predictions. The method is specified in the thesis notebook but is not yet trained or registered. '+
    'The bands below still show how the unqualified forecasts performed.');
  setText('qualification-note',q.legacy_unannotated?q.legacy_unannotated+' older forecasts have no saved issue-time gate and are excluded from coverage.':'');
}

/* Vital signs for the collector, read at a glance. Every value here is measured:
   nothing is shown as healthy unless the API reported it so. */
function tile(id,value,detail,state){
  const node=$(id); if(!node)return;
  node.dataset.state=state||'idle';
  node.querySelector('.hc-value').textContent=value;
  node.querySelector('.hc-detail').textContent=detail;
}
/* Elapsed and remaining time are clocks, not readings: they are recomputed every
   second so the strip is never quietly showing a 30-second-old figure. Run
   duration is a completed measurement and deliberately does not tick. */
const clock={candle:null,next:null,stale:false,candleLabel:'',nextLabel:''};
function elapsed(ms){
  const total=Math.max(0,Math.round(ms/1000));
  if(total<60)return total+'s';
  const m=Math.floor(total/60), sec=total%60;
  if(m<60)return m+'m '+String(sec).padStart(2,'0')+'s';
  return Math.floor(m/60)+'h '+String(m%60).padStart(2,'0')+'m';
}
function tickClocks(){
  if(clock.candle==null)tile('hc-freshness','—',clock.candleLabel||'Awaiting market data','idle');
  else tile('hc-freshness',elapsed(Date.now()-clock.candle),clock.candleLabel,clock.stale?'bad':'ok');

  if(clock.next==null)tile('hc-next','—',clock.nextLabel||'Not scheduled','idle');
  else{
    const left=clock.next-Date.now();
    tile('hc-next',left<=0?'due now':'in '+elapsed(left),clock.nextLabel,left<-300000?'warn':'ok');
  }
}
setInterval(()=>{if(!document.hidden&&clock.candle!=null)tickClocks();},1000);

function renderHealthcheck(data,sys){
  const busy=Boolean(sys.active_run?.id);
  tile('hc-worker',busy?'BUSY':sys.worker_alive?'ONLINE':'OFFLINE',
    busy?'A refresh is running':sys.worker_alive?'Reporting in':'Not reporting',
    busy?'warn':sys.worker_alive?'ok':'bad');

  const runs=sys.runs||[], lastRun=runs[0];
  const failed=lastRun&&['error','failed'].includes(lastRun.status);
  tile('hc-cycle',lastRun?(failed?'FAILED':lastRun.status.toUpperCase()):'—',
    lastRun?when(lastRun.started_at)+' UTC':'No runs recorded',
    !lastRun?'idle':failed?'bad':'ok');

  clock.candle=Date.parse(sys.latest_candle_close)||null;
  clock.next=Date.parse(sys.worker?.next_run_at)||null;
  clock.stale=Boolean(sys.data_stale);
  clock.candleLabel=sys.latest_candle_close?'Since '+when(sys.latest_candle_close)+' UTC':'Awaiting market data';
  clock.nextLabel=sys.worker?.next_run_at?when(sys.worker.next_run_at)+' UTC':'Not scheduled';
  tickClocks();

  const timed=runs.find(r=>r.duration_ms!=null);
  tile('hc-duration',timed?(timed.duration_ms/1000).toFixed(1)+' s':'—','Last completed run',timed?'ok':'idle');

  const newsHours=sys.quality?.hours_with_news_in_latest_sequence;
  tile('hc-sentiment',newsHours==null?'—':newsHours+' / 48','Hours with news in the window',
    newsHours==null?'idle':newsHours===0?'bad':newsHours<12?'warn':'ok');

  tile('hc-scored',data.scored.toLocaleString(),
    data.pending?data.pending+' awaiting outcome':'Outcomes evaluated',data.scored?'ok':'idle');

  setText('hc-updated','UPDATED '+when(data.generated_at)+' UTC');
}

/* Held-out performance, as measured at training time on data the model never
   saw. ROC-AUC leads because accuracy alone hides a degenerate classifier: a
   model that rarely predicts UP can still score ~50% while ranking at chance.
   Nothing here is coloured green unless it actually beats chance. */
function renderHoldout(forecasts){
  const host=$('holdout-grid'); if(!host||!forecasts)return;
  const rows=[]; let worst=null;
  for(const key of ['1','6','24']){
    const f=forecasts[key]; if(!f)continue;
    const m=f.historical_holdout_metrics||{};
    const auc=typeof m.roc_auc==='number'?m.roc_auc:null;
    if(auc!=null)worst=worst==null?auc:Math.min(worst,auc);
    // 0.50 is the chance line for ROC-AUC; a small margin above it is not skill
    const grade=auc==null?'idle':auc>=0.55?'ok':auc>=0.52?'warn':'bad';
    const card=element('article',null,'holdout-row');
    card.dataset.state=grade;
    const head=element('div',null,'holdout-head');
    head.append(element('strong',key==='1'?'1 hour':key+' hours'),
      element('span',modelName(f.model)+' · '+featureName(f.feature_set),'holdout-model'));
    const stats=element('div',null,'holdout-stats');
    for(const [label,value,lead] of [
      ['ROC-AUC',auc==null?'—':auc.toFixed(3),true],
      ['Accuracy',typeof m.accuracy==='number'?percent(m.accuracy):'—',false],
      ['MCC',typeof m.mcc==='number'?(m.mcc>=0?'+':'')+m.mcc.toFixed(3):'—',false]]){
      const cell=element('div',null,lead?'holdout-stat lead':'holdout-stat');
      cell.append(element('span',label),element('strong',value));
      stats.append(cell);
    }
    card.append(head,stats);
    rows.push(card);
  }
  host.replaceChildren(...rows);
  setText('holdout-note', worst==null
    ? 'Held-out metrics are unavailable for the deployed artifacts.'
    : worst<=0.52
      ? 'ROC-AUC at or near 0.50 means the model ranks no better than chance on data it never saw, so the live figure above should not yet be read as skill. Accuracy near 50% on a near-balanced target says the same thing.'
      : 'ROC-AUC above 0.50 indicates the model ranks better than chance on unseen data. Live results remain the decisive test.');
}

function renderSummary(data){
  renderQualification(data.qualification);

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
  $('skill-delta-wrap').classList.toggle('positive',delta!=null&&delta>0);
  $('skill-delta-wrap').classList.toggle('negative',delta!=null&&delta<0);
  const n=data.scored;
  const se=n?Math.sqrt(0.25/n)*100:null;
  setText('sample-note', !n
    ? 'No scored outcomes yet, so the figure above is not yet measurable.'
    : `Based on ${n} scored forecast${n===1?'':'s'} · standard error ±${se.toFixed(1)} pp. `+
      (data.accuracy!=null && Math.abs(data.accuracy-0.5)*100 < 2*se
        ? 'That interval still includes 50%, so this is not yet distinguishable from chance.'
        : 'The gap from 50% is larger than two standard errors on this sample.'));
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
  renderHealthcheck(data,sys);
  if(sys.last_evaluation_error)notify('Outcome evaluation: '+sys.last_evaluation_error,true);
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
  const style=getComputedStyle(document.documentElement);
  const tone=name=>style.getPropertyValue(name).trim();
  const muted=tone('--muted')||'#607c8c',grid=tone('--border')||'#293a44';
  const accent=tone('--cyan')||'#28c4d4',baseline=muted||'#8586b4';
  const left=43,right=box.width-15,top=20,bottom=box.height-29;
  for(let i=0;i<5;i++){const v=1-i/4,y=top+(bottom-top)*i/4;ctx.fillStyle=muted;ctx.fillText(Math.round(v*100)+'%',4,y+3);ctx.beginPath();ctx.setLineDash([3,5]);ctx.strokeStyle=grid;ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();}
  const series=state.series,n=series.length,x=i=>left+(right-left)*(n<=1?.5:i/(n-1)),y=v=>bottom-v*(bottom-top);
  let valid=0;
  for(const [key,color,dash] of [['baseline_accuracy',baseline,[4,4]],['accuracy',accent,[]]]){
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=key==='accuracy'?2:1.3;ctx.setLineDash(dash);ctx.beginPath();let drawing=false;
    series.forEach((p,i)=>{if(p[key]==null){drawing=false;return;}if(key==='accuracy')valid++;if(!drawing)ctx.moveTo(x(i),y(p[key]));else ctx.lineTo(x(i),y(p[key]));drawing=true;});ctx.stroke();ctx.setLineDash([]);
    series.forEach((p,i)=>{if(p[key]!=null){ctx.beginPath();ctx.arc(x(i),y(p[key]),key==='accuracy'?2.7:1.8,0,Math.PI*2);ctx.fill();}});
  }
  ctx.fillStyle=muted;ctx.textAlign='center';
  const tickIndices=[...new Set([0,Math.round((n-1)/3),Math.round((n-1)*2/3),n-1])];
  for(const i of tickIndices){if(series[i])ctx.fillText(new Intl.DateTimeFormat('en-GB',{timeZone:'UTC',month:'short',day:'numeric'}).format(new Date(series[i].date+'T00:00:00Z')),x(i),box.height-7);}
  $('chart-empty').hidden=valid>0;
  canvas.setAttribute('aria-label',valid?'Rolling '+state.days+'-day accuracy and always UP baseline; current accuracy '+percent(series[n-1]?.accuracy):'No scored forecasts yet. Accuracy appears after target candles close.');
}

async function load(){
  // Monitoring rows exist only for an asset with a deployed model; never fill
  // these panels from another asset's history.
  if(!window.CryptoLens.deployed)return;
  const generation=++state.generation;
  try{
    const [summary,history]=await Promise.all([readJson('/monitoring/summary?'+query()),readJson('/monitoring/history?'+query(true))]);
    if(generation!==state.generation)return;
    renderSummary(summary);renderHistory(history);
  }catch(error){if(generation!==state.generation)return;notify('Unable to update from the API. Displayed values may be stale. Retrying automatically.',true);setText('connection-label','API unavailable');$('connection-dot').classList.add('error');setText('last-updated','Connection lost · displayed data may be stale');}
}
window.CryptoLens.on('horizon',horizon=>{
  state.horizon=horizon;state.version='';state.offset=0;load();
});
window.CryptoLens.on('asset',()=>{state.version='';state.offset=0;load();});
window.CryptoLens.on('holdout',forecasts=>renderHoldout(forecasts));
if(window.CryptoLens.holdout)renderHoldout(window.CryptoLens.holdout);
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
window.addEventListener('resize',drawChart);
window.addEventListener('cryptolens:view',drawChart);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)load();});
drawChart();load();
let ticks=0;setInterval(()=>{ticks++;if(!document.hidden&&(state.active||ticks%6===0))load();},5000);
