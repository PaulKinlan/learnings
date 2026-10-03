import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {launch} from './lib/cdp.mjs';
import {serve} from '../scripts/serve.mjs';
const local=process.env.TEST_URL?null:await serve(),base=process.env.TEST_URL??local.url;
const out=resolve(process.env.EVIDENCE_DIR??'../cap-evidence/learnings/browser');await mkdir(out,{recursive:true});
const p=await launch({width:1440,height:1000}),checks=[];
const check=(n,c)=>{assert.ok(c,n);checks.push(n);};
async function key(key){for(const type of ['keyDown','keyUp'])await p.send('Input.dispatchKeyEvent',{type,key});}
async function select(id,index){await p.click('#'+id);await key('Home');for(let i=0;i<index;i++)await key('ArrowDown');await key('Enter');}
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
 await p.emulateViewport({width:390,height:844,mobile:true,scale:1});check('lab no mobile horizontal overflow',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await p.screenshot(out+'/mobile-lab.png',{fullPage:true});
 const resources=await p.evaluate(()=>performance.getEntriesByType('resource').map(x=>x.name));check('reading and local mode load only same-origin assets',resources.every(u=>new URL(u).origin===new URL(base).origin));
 await p.goto(base+'decision-models/adaptation.html');await p.waitFor(()=>document.querySelector('#labelled-examples').value.length>0);
 await p.evaluate(()=>{
  window.__MOCK_DECIDE__=(spec,cfg)=>{
   return {data:{model:'Laya multilingual',answers:{category:{type:'choice',choice:'veln',confidence:0.85,probabilities:{veln:0.85,sova:0.05,tarn:0.05,unclear:0.05}}}},elapsed:12,source:'Laya (LiteRT.js in-browser)',requestedModel:'laya-multilingual'};
  };
 });
 await p.click('#compare');await p.waitFor(()=>document.querySelector('#comparison tbody')!==null);
 check('paired adaptation displays two arms',await p.evaluate(()=>document.querySelectorAll('#comparison tbody tr').length===2));
 check('adaptation no document overflow',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
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
 await p.emulateViewport({width:1440,height:1000,mobile:false,scale:1});
 await p.goto(base+'decision-models/');check('report explicit CORS and browser-inference limitations',await p.evaluate(()=>document.body.textContent.includes('HTTP 400')&&document.body.textContent.includes('Run it in this tab')));
 await p.emulateViewport({width:1440,height:1000,mobile:false,scale:1});await p.screenshot(out+'/desktop-report.png',{fullPage:true});
 const receipt={at:new Date().toISOString(),base,checks,resources,qualification:'UI behavior and synthetic transport fixtures only. No paid model inference, local Kev weights, or training run.'};await writeFile(out+'/receipt.json',JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt,null,2));
}finally{await p.close();if(local)await new Promise(r=>local.server.close(r));}
