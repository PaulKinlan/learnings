// A browser gate on a shared VM must clean up: an orphaned profile blocks the next lane.
// The CDP launcher owns signal/exit cleanup as well as the normal finally below.
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import {basename,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {launch} from './lib/cdp.mjs';
import {serve} from '../scripts/serve.mjs';
const repoRoot=fileURLToPath(new URL('..',import.meta.url));
const commit=execFileSync('git',['rev-parse','--short=12','HEAD'],{cwd:repoRoot,encoding:'utf8'}).trim();
const defaultEvidence=resolve(repoRoot,'..','cap-evidence','learnings',`${basename(repoRoot)}-${commit}`,'browser');
const local=process.env.TEST_URL?null:await serve(),base=process.env.TEST_URL??local.url;
const out=resolve(process.env.EVIDENCE_DIR??defaultEvidence);await mkdir(out,{recursive:true});
const p=await launch({width:1440,height:1000}),checks=[];
const check=(n,c)=>{assert.ok(c,n);checks.push(n);};
async function select(id,index){await p.evaluateWithGesture((selId,idx)=>{const el=document.getElementById(selId);el.focus();el.selectedIndex=idx;el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));},id,index);}
try{
 await p.goto(base+'decision-models/lab.html');await p.waitFor(()=>document.querySelector('#demo-title').textContent.length>0);
 check('default provider is Laya',await p.evaluate(()=>document.querySelector('#provider').value==='laya'&&document.querySelector('#provider-note').textContent.includes('LiteRT.js')));
 await p.evaluate(()=>{
  window.__MOCK_DECIDE__=(spec,cfg)=>{
   const answers={};
   for(const [id,q] of Object.entries(spec.questions)){
    if(q.type==='choice'){
     const keys=Object.keys(q.criteria);
     answers[id]={type:'choice',choice:keys[0],confidence:0.85,probabilities:Object.fromEntries(keys.map((k,i)=>[k,i===0?0.85:0.15/(keys.length-1)]))};
    }else if(q.type==='score'){
     answers[id]={type:'score',score:1.5,confidence:0.85,probabilities:Object.fromEntries(q.criteria.map((_,i)=>[String(i),1/q.criteria.length]))};
    }else{
     answers[id]={type:'noul',noul:0.85,confidence:0.85};
    }
   }
   return {data:{model:'Laya multilingual',answers},elapsed:12,source:'Laya (LiteRT.js in-browser)',requestedModel:'laya-multilingual'};
  };
 });
 await p.click('#run');await p.waitFor(()=>document.querySelector('#answers').children.length>0);
 check('default confidence gate accepts high confidence',await p.evaluate(()=>!document.querySelector('#apply').disabled));
 await p.click('#apply');check('tool choice applies a visible local simulation',await p.evaluate(()=>document.querySelector('#effect').textContent.includes('Simulated list_tabs')));
 for(const [i,label] of [[1,'routing'],[2,'adaptive'],[3,'moderation'],[4,'score'],[5,'ranking'],[6,'machine'],[7,'game']]){
  await select('example',i);await p.click('#run');await p.waitFor(()=>document.querySelector('#answers').children.length>0);
  check(label+' has typed outputs and provenance',await p.evaluate(()=>document.querySelector('#status').textContent.includes('Laya')&&document.querySelectorAll('meter').length>=2));
  if([1,2,6,7].includes(i)){await p.click('#apply');check(label+' applies visible behavior',await p.evaluate(()=>!document.querySelector('#effect').textContent.includes('No action applied')));}
  if(i===2)check('adaptive UI inserts trusted input',await p.evaluate(()=>Boolean(document.querySelector('#adaptive-answer'))));
  if(i===6)check('state machine progresses to triaged',await p.evaluate(()=>document.querySelector('#world').textContent.includes('triaged')));
  if(i===7)check('game visibly moves',await p.evaluate(()=>document.querySelector('.grid').getAttribute('aria-label').includes('column 1')));
 }
 await p.type('#state','Changed state');check('editing invalidates previous actions',await p.evaluate(()=>document.querySelector('#apply').disabled&&document.querySelector('#status').textContent.includes('Inputs changed')));
 await p.click('#seed');await p.click('#load-spec');check('checked library example loads without a key',await p.evaluate(()=>document.querySelector('#generation-status').textContent.includes('Loaded safely')));
 await p.type('#specification','{"title":"x","script":"alert(1)"}');await p.click('#load-spec');check('executable field rejected',await p.evaluate(()=>document.querySelector('#generation-status').textContent.includes('refused')));
 const spec={title:'<img src=x onerror=alert(1)>',description:'Plain text, not HTML',state:'test',questions:{question:{type:'noul',instructions:'Is this a test?'}}};
 await p.type('#specification',JSON.stringify(spec));await p.click('#load-spec');check('HTML-looking strings remain inert text',await p.evaluate(()=>!document.querySelector('#demo-title img')&&document.querySelector('#demo-title').textContent.startsWith('<img')));
 // Browser transport fixtures exercise generation plumbing, not paid provider quality.
 await p.evaluate(()=>{window.__fetch=window.fetch;window.fetch=async(url,options)=>({ok:true,json:async()=>url.includes('anthropic')?{content:[{type:'text',text:JSON.stringify({title:'Claude fixture',description:'Synthetic transport fixture',state:'test',questions:{q:{type:'noul',instructions:'Is this a test?'}}})}]}:{choices:[{message:{content:JSON.stringify({title:'OpenAI fixture',description:'Synthetic transport fixture',state:'test',questions:{q:{type:'noul',instructions:'Is this a test?'}}})}}]}});});
 await p.type('#generator-key','synthetic-test-not-a-real-key');await p.click('#generate');await p.waitFor(()=>document.querySelector('#generation-status').textContent.startsWith('Generated and'));
 check('OpenAI generation path validates fixture response',await p.evaluate(()=>document.querySelector('#specification').value.includes('OpenAI fixture')));
 await select('generator-provider',1);await p.click('#generate');await p.waitFor(()=>document.querySelector('#generation-status').textContent.startsWith('Generated and'));
 check('Claude generation path validates fixture response',await p.evaluate(()=>document.querySelector('#specification').value.includes('Claude fixture')));
 await p.evaluate(()=>{window.fetch=window.__fetch;delete window.__fetch;});
 await p.click('#clear-keys');check('keys cleared and never saved',await p.evaluate(()=>document.querySelector('#generator-key').value===''&&localStorage.length===0));
 await select('provider',2);await p.click('#run');await p.waitFor(()=>document.querySelector('#status').textContent.includes('key first'));check('missing Jev key fails before request',true);
 await select('provider',0);await select('example',0);await p.click('#run');await p.waitFor(()=>document.querySelector('#answers').children.length>0);
 await p.evaluate(()=>scrollTo(0,0));await p.screenshot(out+'/desktop-lab.png',{fullPage:true});
 await p.emulateViewport({width:390,height:844,mobile:true,scale:1});check('lab no mobile horizontal overflow',await p.evaluate(()=>innerWidth===390&&document.documentElement.scrollWidth<=innerWidth));await p.screenshot(out+'/mobile-lab.png',{fullPage:true});
 const resources=await p.evaluate(()=>performance.getEntriesByType('resource').map(x=>x.name));check('reading and local mode load only same-origin assets',resources.every(u=>new URL(u).origin===new URL(base).origin));
 await p.goto(base+'decision-models/adaptation.html');await p.waitFor(()=>document.querySelector('#labelled-examples').value.length>0);
 await p.evaluate(()=>{
  window.__MOCK_DECIDE__=(spec,cfg)=>{
   return {data:{model:'Laya multilingual',answers:{category:{type:'choice',choice:'veln',confidence:0.85,probabilities:{veln:0.85,sova:0.05,tarn:0.05,unclear:0.05}}}},elapsed:12,source:'Laya (LiteRT.js in-browser)',requestedModel:'laya-multilingual'};
  };
 });
 await p.click('#compare');await p.waitFor(()=>document.querySelector('#comparison tbody')!==null);
 check('paired adaptation displays two arms',await p.evaluate(()=>document.querySelectorAll('#comparison tbody tr').length===2));
 check('adaptation no document overflow',await p.evaluate(()=>innerWidth===390&&document.documentElement.scrollWidth<=innerWidth));
 await p.screenshot(out+'/mobile-adaptation.png',{fullPage:true});
 await p.goto(base+'decision-models/playground.html');await p.waitFor(()=>document.querySelector('#engine')!==null);
 check('playground offers Laya, Kev and Jev',await p.evaluate(()=>Array.from(document.querySelectorAll('#engine option')).map(o=>o.value).join(',')==='laya,kev,jev'));
 check('playground has window.Classifier polyfill installed',await p.evaluate(()=>typeof window.Classifier==='function'));
 await p.screenshot(out+'/desktop-playground.png',{fullPage:true});
 await p.goto(base+'decision-models/architecture.html');await p.waitFor(()=>document.querySelector('#cost-shape').textContent.includes('Illustrative'));const before=await p.evaluate(()=>document.querySelector('#cost-shape').textContent);await p.type('#tokens','200');check('architecture calculator responds to input',await p.evaluate(()=>document.querySelector('#cost-shape').textContent)!==before);
 await p.goto(base+'decision-models/catalogue.html');check('catalogue contains 43 concrete entries',await p.evaluate(()=>document.querySelectorAll('tbody tr').length===43));check('catalogue no mobile document overflow',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await p.goto(base+'decision-models/image-lab.html');await p.waitFor(()=>document.querySelector('#bench-table-container table')!==null);
 check('image lab loads default UI preset',await p.evaluate(()=>document.querySelector('#preview-image').src.startsWith('data:image/')&&document.querySelector('#questions').value.includes('UI Error Modal')));
 check('jev image bench table renders 50 systems',await p.evaluate(()=>document.querySelectorAll('#bench-table-container tbody tr').length===50));
 await p.click('#preset-invoice');
 check('preset selection switches to invoice questions',await p.evaluate(()=>document.querySelector('#questions').value.includes('Invoice Legibility')));
 await p.click('#preset-navigation');
 check('preset selection switches to navigation questions',await p.evaluate(()=>document.querySelector('#questions').value.includes('Visual Navigation')));
 await p.click('#preset-ui');
 await p.click('#run-decision');await p.waitFor(()=>document.querySelector('#answers').children.length>0);
 check('image decision evaluates questions with distribution meters',await p.evaluate(()=>document.querySelectorAll('#answers meter').length>=7&&document.querySelector('#status').textContent.includes('Decision complete')));
 check('image decision renders gate verdicts',await p.evaluate(()=>document.querySelectorAll('#answers .gate').length>=3));
 check('demo engine renders explicit simulation disclosure badge',await p.evaluate(()=>document.querySelector('#simulation-badge')!==null&&document.querySelector('#simulation-badge').textContent.includes('Notice: UI and schema validation simulation')));
 check('engine notice discloses simulation when client engine selected',await p.evaluate(()=>document.querySelector('#engine-notice')!==null&&document.querySelector('#engine-notice').textContent.includes('Notice: UI and schema validation simulation')));
 await p.evaluate(()=>{
  const dt=new DataTransfer();
  const file=new File(['dummy-bytes'],'test-upload.png',{type:'image/png'});
  dt.items.add(file);
  const input=document.querySelector('#image-file');
  input.files=dt.files;
  input.dispatchEvent(new Event('change',{bubbles:true}));
 });
 await p.waitFor(()=>document.querySelector('#status').textContent.includes('test-upload.png'));
 check('file upload updates preview and status',await p.evaluate(()=>document.querySelector('#status').textContent.includes('test-upload.png')));
 await p.type('#bench-search','Qwen');
 check('benchmark table search filters systems',await p.evaluate(()=>document.querySelectorAll('#bench-table-container tbody tr').length<50&&document.querySelectorAll('#bench-table-container tbody tr').length>=5));
 await p.screenshot(out+'/desktop-image-lab.png',{fullPage:true});
 await p.emulateViewport({width:390,height:844,mobile:true,scale:1});
 check('image lab no mobile horizontal overflow',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await p.screenshot(out+'/mobile-image-lab.png',{fullPage:true});
 await p.emulateViewport({width:390,height:844,mobile:true,scale:1});
 const blockMathOnOneLine=()=>[...document.querySelectorAll('main math[display="block"]')].every((m)=>{const ks=[...m.children].map((k)=>k.getBoundingClientRect()).filter((r)=>r.height>0);return getComputedStyle(m).display==='block math'&&Math.max(...ks.map((r)=>r.top))<Math.min(...ks.map((r)=>r.bottom));});
 await p.send('Page.addScriptToEvaluateOnNewDocument',{source:`window.__nnErrors=[];addEventListener('error',e=>window.__nnErrors.push(e.message));addEventListener('unhandledrejection',e=>window.__nnErrors.push(String(e.reason)));addEventListener('securitypolicyviolation',e=>window.__nnErrors.push(e.violatedDirective+': '+e.blockedURI));`});
 await p.goto(base+'neural-networks/');await p.waitFor(()=>document.querySelectorAll('#nn-stack > li[data-slug]').length>0);
 check('neural-networks hub: every block formula lays out on one line (display: block math)',await p.evaluate(blockMathOnOneLine));
 check('neural-networks chapter hub renders the chapter stack and native MathML without overflow',await p.evaluate(()=>document.querySelectorAll('#nn-stack > li[data-slug]').length>=15&&document.querySelectorAll('main math').length>=3&&innerWidth===390&&document.documentElement.scrollWidth<=innerWidth));
 // Each new playground is exercised, not just counted. Native keyboard input drives sliders.
 const widgetControl=(kind,key)=>`[data-widget="${kind}"] [data-control="${key}"]`;
 const widgetAction=(kind,key)=>`[data-widget="${kind}"] [data-action="${key}"]`;
 async function chooseWidget(kind,key,value){await p.evaluate((sel,v)=>{const input=document.querySelector(sel);input.value=v;input.dispatchEvent(new Event('change',{bubbles:true}));},widgetControl(kind,key),value);}
 async function rangeKey(kind,key,name){await p.click(widgetControl(kind,key));await p.send('Input.dispatchKeyEvent',{type:'keyDown',key:name,code:name});await p.send('Input.dispatchKeyEvent',{type:'keyUp',key:name,code:name});}
 async function widgetScreenshot(kind,name=kind){
  await p.emulateViewport({width:1440,height:1000,mobile:false,scale:1});
  const clip=await p.evaluate(k=>{const r=document.querySelector(`[data-widget="${k}"]`).getBoundingClientRect();return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1};},kind);
  const {data}=await p.send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip});
  await writeFile(out+`/nn-${name}-interaction.png`,Buffer.from(data,'base64'));
  await p.emulateViewport({width:390,height:844,mobile:true,scale:1});
 }
 check('all sixteen chapter cards are active links',await p.evaluate(()=>document.querySelectorAll('#nn-stack h3 a').length===16&&!document.querySelector('#nn-stack').textContent.includes('planned')));
 await p.click('#nn-stack li[data-slug="neuron-and-perceptron.html"] p');
 await p.waitFor(()=>location.pathname.endsWith('neuron-and-perceptron.html')&&document.title.includes('The neuron and the perceptron'));
 check('chapter card body click navigates to chapter page', await p.evaluate(()=>location.pathname.endsWith('neuron-and-perceptron.html')));
 await p.goto(base+'neural-networks/');
 await p.waitFor(()=>document.querySelectorAll('#nn-stack > li[data-slug]').length>0);
 await p.evaluate(()=>{const c=document.querySelector('[data-widget="perceptron"] canvas'),ctx=c.getContext('2d'),original=ctx.lineTo;ctx.lineTo=function(x,y){c.__boundary=[x,y];return original.call(this,x,y);};});
 await rangeKey('perceptron','w1','Home');
 const boundaryBefore=await p.evaluate(()=>document.querySelector('[data-widget="perceptron"] canvas').__boundary);
 await rangeKey('perceptron','w1','End');
 check('perceptron slider changes actual canvas boundary coordinates and numeric equation',await p.evaluate(old=>{const el=document.querySelector('[data-widget="perceptron"]');return el.querySelector('canvas').__boundary[1]!==old[1]&&el.querySelector('[data-status]').textContent.includes('Boundary: 4x₁');},boundaryBefore));
 await chooseWidget('perceptron','preset','nand');
 const weightsBefore=await p.evaluate(()=>['w1','w2','b'].map(k=>document.querySelector(`[data-widget="perceptron"] [data-control="${k}"]`).value));
 await p.click(widgetAction('perceptron','step'));await p.click(widgetAction('perceptron','step'));await p.click(widgetAction('perceptron','step'));
 check('perceptron training updates both weights and bias in visible controls',await p.evaluate(old=>{const values=['w1','w2','b'].map(k=>document.querySelector(`[data-widget="perceptron"] [data-control="${k}"]`).value);return values.every((v,i)=>v!==old[i]);},weightsBefore));
 await chooseWidget('perceptron','preset','xor');await p.click(widgetAction('perceptron','step'));
 check('XOR has mistakes and an explicit impossibility proof',await p.evaluate(()=>{const el=document.querySelector('[data-widget="perceptron"]');return !el.querySelector('[data-status]').textContent.startsWith('4/4')&&el.querySelector('[data-proof]').textContent.includes('contradiction');}));
 await widgetScreenshot('perceptron');
 await chooseWidget('landscape','optimizer','adam');
 const lossBefore=await p.evaluate(()=>document.querySelector('[data-widget="landscape"] [data-status]').textContent);
 await p.click(widgetAction('landscape','run'));
 await p.waitFor(()=>/Step ([2-9]|\d{2,}):/.test(document.querySelector('[data-widget="landscape"] [data-status]').textContent));
 await p.click(widgetAction('landscape','stop'));
 check('selected Adam optimizer advances the trajectory and changes coordinates and loss',await p.evaluate(old=>{const s=document.querySelector('[data-widget="landscape"] [data-status]').textContent;return s!==old&&!s.includes('(1.8, 1.5)')&&!s.includes('loss 12.87');},lossBefore));
 await widgetScreenshot('landscape');
 await chooseWidget('landscape','surface','doubleWell');
 check('landscape change clears a previous path and explains local minima',await p.evaluate(()=>document.querySelector('[data-widget="landscape"] [data-status]').textContent.includes('Step 0:')&&document.querySelector('[data-widget="landscape"] [data-status]').textContent.includes('shallow local')));
 check('backprop range controls retain fractional parameter defaults',await p.evaluate(async()=>{const {TINY_NET_DEFAULTS}=await import('./math.js');return Object.entries(TINY_NET_DEFAULTS.params).every(([key,value])=>Number(document.querySelector(`[data-widget="backprop"] [data-control="${key}"]`).value)===value);}));
 await p.click(widgetAction('backprop','forward'));
 check('backprop forward click reveals a numerical node activation',await p.evaluate(()=>document.querySelector('[data-widget="backprop"] .active-node strong').textContent.match(/= -?\d/)!==null));
 await widgetScreenshot('backprop','backprop-forward');
 await p.evaluate(()=>{const b=document.querySelector('[data-widget="backprop"] [data-action="forward"]');while(!b.disabled)b.click();});
 await p.click(widgetAction('backprop','backward'));
 check('backward step displays exact local chain-rule multiplication',await p.evaluate(()=>document.querySelector('[data-widget="backprop"] [data-status]').textContent.includes('×')));
 await widgetScreenshot('backprop','backprop-backward');
 await p.evaluate(()=>{const b=document.querySelector('[data-widget="backprop"] [data-action="backward"]');while(!b.disabled)b.click();});
 check('backprop reaches nonzero dL/dw11 matching an independent finite difference',await p.evaluate(async()=>{const m=await import('./math.js');const node=[...document.querySelectorAll('[data-widget="backprop"] .graph-node')].find(n=>n.querySelector('strong').textContent.startsWith('w11 ='));const actual=Number(node.querySelector('p:last-child').textContent.match(/= ([^;]+)/)[1]);const f=w=>m.buildTinyNetwork({...m.TINY_NET_DEFAULTS,params:{...m.TINY_NET_DEFAULTS.params,w11:w}}).loss.data;const expected=(f(.50001)-f(.49999))/.00002;return actual!==0&&Math.abs(actual-expected)<.00006;}));
 await rangeKey('backprop','x1','Home');
 check('editing a graph input invalidates stale forward values and gradients',await p.evaluate(()=>document.querySelector('[data-widget="backprop"] [data-action="backward"]').disabled&&[...document.querySelectorAll('[data-widget="backprop"] .graph-node strong')].every(n=>n.textContent.endsWith('?'))));
 await p.click(widgetAction('convolution','step'));
 const fieldBefore=await p.evaluate(()=>[...document.querySelectorAll('[data-widget="convolution"] .receptive')].map(b=>b.getAttribute('aria-label')).join('|'));
 await p.click(widgetAction('convolution','step'));
 check('CNN step moves nine highlighted pixels and writes the second output cell',await p.evaluate(old=>{const el=document.querySelector('[data-widget="convolution"]');return [...el.querySelectorAll('.receptive')].map(b=>b.getAttribute('aria-label')).join('|')!==old&&el.querySelectorAll('.receptive').length===9&&el.querySelectorAll('[data-output] td')[1].textContent!=='·'&&el.querySelectorAll('[data-output] td')[2].textContent==='·';},fieldBefore));
 await widgetScreenshot('convolution');
 await chooseWidget('convolution','padding','1');await chooseWidget('convolution','stride','2');
 check('CNN stride and padding recalculate output dimensions and clear the scan',await p.evaluate(()=>document.querySelector('[data-widget="convolution"] [data-status]').textContent.startsWith('4 × 4')&&[...document.querySelectorAll('[data-widget="convolution"] [data-output] td')].every(td=>td.textContent==='·')));
 await p.click('[data-widget="convolution"] [data-image] button');
 check('CNN pixel input is editable with a native button',await p.evaluate(()=>document.querySelector('[data-widget="convolution"] [data-image] button').getAttribute('aria-pressed')==='true'));
 const probBefore=await p.evaluate(()=>({p:document.querySelector('[data-widget="probabilities"] meter').value,s:document.querySelector('[data-widget="probabilities"] [data-status]').textContent}));
 await rangeKey('probabilities','tau','End');
 check('temperature changes visible probabilities and increases entropy while preserving sum one',await p.evaluate(old=>{const el=document.querySelector('[data-widget="probabilities"]'),s=el.querySelector('[data-status]').textContent;const entropy=t=>Number(t.match(/Entropy = ([\d.]+)/)[1]);return el.querySelector('meter').value<old.p&&el.querySelector('output').textContent!==''&&entropy(s)>entropy(old.s)&&Math.abs([...el.querySelectorAll('meter')].reduce((a,m)=>a+m.value,0)-1)<1e-12;},probBefore));
 await widgetScreenshot('probabilities');
 await chooseWidget('attention','token','1');
 const attentionBefore=await p.evaluate(()=>document.querySelector('[data-widget="attention"] [data-vectors]').textContent);
 await chooseWidget('attention','mask','causal');
 await p.click('[data-widget="attention"] tbody tr:nth-child(2) td:nth-of-type(6) button');
 check('attention mask zeroes future-token cells and updates weighted V output',await p.evaluate(old=>{const el=document.querySelector('[data-widget="attention"]');return el.querySelector('[data-status]').textContent.includes('masked to −∞')&&el.querySelector('[data-status]').textContent.includes('weight = 0')&&el.querySelector('[data-vectors]').textContent!==old;},attentionBefore));
 await widgetScreenshot('attention');
 await p.click('[data-widget="landscape"] canvas');
 check('clicking the landscape drops a new start point and clears the old trajectory',await p.evaluate(()=>{const el=document.querySelector('[data-widget="landscape"]');return el.querySelector('[data-control="x"]').value!=='1.8'&&el.querySelector('[data-status]').textContent.startsWith('Step 0:');}));
 await p.send('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
 await p.click(widgetAction('landscape','run'));await p.waitFor(()=>document.querySelector('[data-widget="landscape"] [data-status]').textContent.startsWith('Step 160:'));
 await p.click(widgetAction('convolution','run'));
 check('reduced-motion mode completes trajectories and convolution without a timed animation',await p.evaluate(()=>document.querySelector('[data-widget="landscape"] [data-status]').textContent.startsWith('Step 160:')&&[...document.querySelectorAll('[data-widget="convolution"] [data-output] td')].every(td=>td.textContent!=='·')));
 await p.send('Emulation.setEmulatedMedia',{features:[]});
 check('hub interactions produce no runtime or CSP errors',await p.evaluate(()=>window.__nnErrors.length===0));
 // Each chapter mounts the same real widgets at its own URL, including nested backend roots.
 const chapterPaths=await p.evaluate(()=>[...document.querySelectorAll('#nn-stack h3 a')].map(a=>a.href));
 for(const url of chapterPaths){await p.goto(url);await p.waitFor(()=>document.querySelector('[data-chapter-nav] a[aria-current="page"]')!==null);check('chapter loads without overflow: '+new URL(url).pathname,await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth&&document.querySelector('main h1').textContent.length>0&&window.__nnErrors.length===0));}
 await p.goto(base+'neural-networks/activations.html');await p.waitFor(()=>document.querySelector('[data-depth] tr')!==null);
 const sigmoidGradient=await p.evaluate(()=>document.querySelector('[data-widget="activations"] [data-status]').textContent);
 await chooseWidget('activations','activation','relu');
 check('activation selection updates the ten-layer gradient calculation',await p.evaluate(old=>{const s=document.querySelector('[data-widget="activations"] [data-status]').textContent;return s!==old&&s.includes('first input: 1.');},sigmoidGradient));
 await p.goto(base+'neural-networks/modern-advancements.html');await p.waitFor(()=>document.querySelector('[data-modern]')!==null);
 const cacheBefore=await p.evaluate(()=>document.querySelector('[data-widget="modern"] [data-status]').textContent);
 await chooseWidget('modern','heads','1');
 check('GQA/MQA control reduces calculated KV-cache size',await p.evaluate(old=>{const size=t=>Number(t.match(/= ([\d.]+) MiB/)[1]);return size(document.querySelector('[data-widget="modern"] [data-status]').textContent)===size(old)/8;},cacheBefore));
 await p.goto(base+'neural-networks/backends/litert.html');await p.waitFor(()=>document.querySelector('[data-quant]')!==null);
 const quantBefore=await p.evaluate(()=>document.querySelector('[data-quant]').textContent);
 await rangeKey('quantization','outlier','End');
 check('LiteRT quantization example changes quantized and decoded values',await p.evaluate(old=>document.querySelector('[data-quant]').textContent!==old,quantBefore));
 for(const kind of ['javascript','webassembly','webgpu']){
  await p.goto(base+`neural-networks/backends/${kind}.html`);await p.waitFor(()=>document.querySelector('[data-widget="backend"] [data-action="run"]')!==null);
  await p.click(widgetAction('backend','run'));await p.waitFor(()=>!document.querySelector('[data-widget="backend"] [data-action="run"]').disabled);
  check(kind+' executes the known GEMM or explicitly reports unavailable GPU',await p.evaluate(k=>{const s=document.querySelector('[data-widget="backend"] [data-status]').textContent;return s.includes(`${k} result: [19, 22, 43, 50]`)||(k==='webgpu'&&s.startsWith('Cannot run:'));},kind));
 }
 await p.goto(base+'neural-networks/gradient-descent.html');await p.waitFor(()=>document.querySelectorAll('#gd-results tr').length===6);
 check('gradient-descent race draws six optimisers, sidebar marks the chapter, MathML renders, no mobile overflow',await p.evaluate(()=>document.querySelectorAll('#gd-results tr').length===6&&document.querySelector('[data-chapter-nav] a[aria-current="page"]')!==null&&document.querySelectorAll('main math').length>=8&&innerWidth===390&&document.documentElement.scrollWidth<=innerWidth));
 check('gradient-descent: every block formula lays out on one line (display: block math)',await p.evaluate(blockMathOnOneLine));
 check('gradient-descent: the learning-rate readout shows the exact race rate (0.15), not the slider rounding (0.151)',await p.evaluate(()=>document.querySelector('#gd-lr-out').textContent==='0.15'));
 await p.click('#gd-compare');await p.waitFor(()=>document.querySelectorAll('#gd-results tr').length===2);
 check('gradient-descent stability demo: eta 0.19 reaches the minimum and 0.21 diverges',await p.evaluate(()=>{const r=[...document.querySelectorAll('#gd-results tr')].map((t)=>t.textContent);return r[0].includes('Reached')&&r[1].includes('Diverged');}));
 await p.goto(base+'neural-networks/lab.html');await p.waitFor(()=>document.querySelector('#mlp-epoch').textContent!=='0');
 check('neural-networks WASM GEMM kernel and interactive trainers render',await p.evaluate(()=>document.querySelector('#kernel-readout').textContent.includes('WebAssembly')&&document.querySelectorAll('#builder-stack .block-item').length===6));
 await p.click('#stepper-next-btn');await p.click('#cnn-step-next-btn');await p.click('#rnn-step-next-btn');await p.click('#tf-step-next-btn');
 check('neural-networks node graph and all 8 section step-by-step debuggers advance micro-steps',await p.evaluate(()=>document.querySelectorAll('#nn-stepper-svg .svg-node').length===11&&document.querySelector('#stepper-stage-title').textContent.includes('Stage 2')&&document.querySelector('#cnn-step-stage-title').textContent.includes('Stage 2')&&document.querySelector('#rnn-step-stage-title').textContent.includes('Stage 2')&&document.querySelector('#tf-step-stage-title').textContent.includes('Stage 2')&&['#nn-stepper-svg','#cnn-step-svg','#rnn-step-svg','#resnet-step-svg','#tf-step-svg','#diff-step-svg','#dec-step-svg','#bld-step-svg'].every(sel=>document.querySelectorAll(sel+' .svg-node').length>=4)));
 await p.click('[data-global-lang="webgpu"]');
 check('neural-networks hyperparameter encyclopedia, RNN BPTT, Deep ResNet, and WebGPU WGSL code explorer render',await p.evaluate(()=>document.querySelectorAll('#mlp-hyperparam-explainer .hyperparam-card').length===5&&document.querySelectorAll('#rnn-temporal-strip .grad-row').length===8&&document.querySelectorAll('#resnet-depth-bars .grad-row').length===12&&document.querySelector('#global-code-display').textContent.includes('@compute @workgroup_size')));
 check('neural-networks no mobile document overflow',await p.evaluate(()=>innerWidth===390&&document.documentElement.scrollWidth<=innerWidth));
 await p.goto(base+'celld/');await p.waitFor(()=>document.querySelector('#cas-bucket-state').textContent.includes('ownership.json'));
 await p.click('#btn-cas-partition');await p.click('#btn-cas-zombie');
 check('celld CAS epoch fencing simulator rejects zombie write',await p.evaluate(()=>document.querySelector('#cas-log').textContent.includes('FENCED 412')));
 check('celld no mobile document overflow',await p.evaluate(()=>innerWidth===390&&document.documentElement.scrollWidth<=innerWidth));
 await p.emulateViewport({width:1440,height:1000,mobile:false,scale:1});
 await p.goto(base+'opt-chronicles/');await p.waitFor(()=>document.querySelectorAll('.event-row').length>10);
 check('opt-chronicles timeline and simulator render under strict CSP',await p.evaluate(()=>document.querySelector('#res-days').textContent.includes('days')));
 // These four harness pages bootstrap through a module script while carrying a strict CSP
 // (`script-src 'self'`). 'self' does not authorise inline scripts, so the bootstrap must be an
 // external same-origin module; when it was inline, the CSP refused it and every page failed
 // SILENTLY — no result global, no exception the suite would ever see. Load each page over http
 // and require its result global to exist as the observable proof that the harness actually ran.
 //
 // POLL, DO NOT SAMPLE: the result global is the module's first observable side effect, and each
 // page goes on fetching real model assets long after it appears. Sampling once ~150ms after load
 // only passed because all four modules happened to assign their global synchronously at top
 // level; it read a module that assigned it after any await as a harness that never ran.
 //
 // INVARIANT (the bound this poll replaces): each decision-models smoke module must assign
 // window.__<name> synchronously at top level, before any await, until that poll is in place.
 // The synchronous assignment is what makes the global proof the harness STARTED under CSP rather
 // than a timing race — a module that awaited first could not be told apart, at the old single
 // sample, from one the CSP had refused.
 //
 // CRASH IS NOT SUCCESS: `present` alone cannot tell a page that wired itself up and then threw
 // from one that is merely still loading. These harnesses fetch real model assets, so "not
 // finished yet" is normal and must pass; only an uncaught error is a crash. Listeners injected
 // into every new document (browser-level, so CSP does not apply to the injection) record uncaught
 // errors and rejections, which catches a module that throws while defining its global.
 //
 // UNCAUGHT ERRORS ARE ALWAYS FATAL. Unhandled rejections are only fatal when they fire before
 // the harness has signalled (set its result global) — a module that rejects at top level and
 // never signals is a crash. A rejection after the signal is a fire-and-forget side effect the
 // harness never awaited, so it is recorded separately as a non-fatal note rather than failing the
 // gate or silently disappearing.
 const harnesses={};
 await p.send('Page.addScriptToEvaluateOnNewDocument',{source:`window.__pageErrors=[];window.__pageNotes=[];addEventListener('error',e=>window.__pageErrors.push(String((e&&e.message)||(e&&e.error)||e)));addEventListener('unhandledrejection',e=>{const text='unhandled rejection: '+String(e&&e.reason);(Boolean(window.__gate||window.__quant||window.__kev||window.__measure)?window.__pageNotes:window.__pageErrors).push(text);});`});
 for(const [page,resultGlobal] of [['gate-smoke.html','__gate'],['quant-smoke.html','__quant'],['kev-smoke.html','__kev'],['measure.html','__measure']]){
  await p.goto(base+'decision-models/'+page);
  try{await p.waitFor((name)=>Boolean(window[name]),{label:`${page} to set window.${resultGlobal}`,timeout:5000,args:[resultGlobal]});}catch{}
  harnesses[page]=await p.evaluate((name)=>{const g=window[name];const pageErrors=(window.__pageErrors??[]).slice(0,5),pageNotes=(window.__pageNotes??[]).slice(0,5);return g?{present:true,stage:g.stage??null,done:g.done??null,error:g.error?String(g.error).slice(0,200):null,pageErrors,pageNotes}:{present:false,pageErrors,pageNotes};},resultGlobal);
  check(`${page} harness raised no uncaught page error${harnesses[page].pageErrors.length?': '+harnesses[page].pageErrors.join(' | '):''}`,harnesses[page].pageErrors.length===0);
  check(`${page} harness executes under CSP and sets window.${resultGlobal}${harnesses[page].pageErrors.length?` (page errors: ${harnesses[page].pageErrors.join(' | ')})`:''}`,harnesses[page].present);
 }
 await p.goto(base+'decision-models/');check('report explicit CORS and browser-inference limitations',await p.evaluate(()=>document.body.textContent.includes('HTTP 400')&&document.body.textContent.includes('Run it in this tab')));
 await p.emulateViewport({width:1440,height:1000,mobile:false,scale:1});await p.screenshot(out+'/desktop-report.png',{fullPage:true});
 const receipt={at:new Date().toISOString(),commit,base,checks,resources,harnesses,qualification:'UI behavior and synthetic transport fixtures only. No paid model inference, local Kev weights, or training run.'};await writeFile(out+'/receipt.json',JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));
}finally{await p.close();if(local)await new Promise(r=>local.server.close(r));}
