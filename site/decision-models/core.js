// Plain data is the execution boundary: generated specifications are never evaluated as code.
const own = (o, k) => Object.hasOwn(o, k);
const record = x => x !== null && typeof x === 'object' && !Array.isArray(x);
const fail = message => { throw new Error(message); };
// decide()/generate() validate their own scalar arguments; the spec walker below reports instead.
const text = (s, name) => { if (typeof s !== 'string' || !s.trim()) fail(`${name} must be non-empty text.`); };
const SPEC_FIELDS = ['title','description','state','questions'];
const QUESTION_FIELDS = ['type','instructions','criteria'];
// One rule walker, two readers. validateSpecIssues() returns every problem with a JSON path for
// the schema playground; validateSpec() throws the first one, so every existing caller keeps the
// exact message and check order it had. A rule added here can never leave the two disagreeing.
function reportKeys(o, allowed, path, add) {
  if (!record(o) || Object.keys(o).some(k => !allowed.includes(k))) {
    add(path, 'Unexpected specification field. Only the documented JSON schema is accepted.');
    return false;
  }
  return true;
}
function reportText(s, name, path, add) {
  if (typeof s !== 'string' || !s.trim()) { add(path, `${name} must be non-empty text.`); return false; }
  return true;
}
export function validateSpecIssues(input) {
  const issues = [];
  const add = (path, message) => { issues.push({ path, message }); };
  if (!reportKeys(input, SPEC_FIELDS, '', add)) return issues;
  reportText(input.title,'Title','title',add); reportText(input.description,'Description','description',add); reportText(input.state,'State','state',add);
  if (!record(input.questions) || !Object.keys(input.questions).length) { add('questions','At least one question is required.'); return issues; }
  for (const [id,q] of Object.entries(input.questions)) {
    const at = `questions.${id}`;
    if (!/^[a-z][a-z0-9_]*$/.test(id) || ['constructor','prototype','__proto__'].includes(id)) add(at,'Use simple lowercase question IDs.');
    if (!reportKeys(q, QUESTION_FIELDS, at, add)) continue;
    reportText(q.instructions,'Instructions',`${at}.instructions`,add);
    if (!['noul','choice','score'].includes(q.type)) { add(`${at}.type`,'Question type must be noul, choice or score.'); continue; }
    if(q.type === 'choice') {
      if(!record(q.criteria) || Object.keys(q.criteria).length < 2 || Object.keys(q.criteria).length > 255) add(`${at}.criteria`,'Choice requires 2–255 named options in this workbench.');
      else for(const [k,v] of Object.entries(q.criteria)) { reportText(k,'Option',`${at}.criteria.${k}`,add); reportText(v,'Option description',`${at}.criteria.${k}`,add); }
    }
    if(q.type === 'score') {
      if(!Array.isArray(q.criteria) || q.criteria.length<2 || q.criteria.length>10) add(`${at}.criteria`,'Score requires 2–10 ordered levels (Jev-compatible).');
      else q.criteria.forEach((v,i)=>reportText(v,'Score level',`${at}.criteria[${i}]`,add));
    }
    if(q.type === 'noul' && own(q,'criteria')) {
      if (reportKeys(q.criteria,['true','false'],`${at}.criteria`,add))
        for(const [k,v] of Object.entries(q.criteria)) reportText(v,'Noul criterion',`${at}.criteria.${k}`,add);
    }
  }
  return issues;
}
export function validateSpec(s) {
  const [first] = validateSpecIssues(s);
  if (first) fail(first.message);
  return structuredClone(s);
}
export function requestFor(spec, model) { validateSpec(spec); text(model,'Model'); return {model,state:spec.state,questions:spec.questions}; }
export function validateAnswers(data, questions) {
  if(!record(data) || !record(data.answers)) fail('Provider did not return an answers object.');
  const unit = v => typeof v==='number' && Number.isFinite(v) && v>=0 && v<=1;
  // The answer-ID set must match the question-ID set exactly. An unexpected answer ID is
  // refused here, before any caller renders a single block, so a provider that returns extra
  // answers fails closed instead of half-rendering the requested ones. (Duplicate wire keys are
  // not detectable here: response.json()/JSON.parse has already collapsed them to the last
  // occurrence — a documented, pinned limitation, see site/decision-models/playground.js.)
  for(const id of Object.keys(data.answers)) {
    if(!own(questions, id)) fail(`Unexpected answer ID: ${id}.`);
  }
  for(const [id,q] of Object.entries(questions)) {
    const a=data.answers[id]; if(!record(a) || a.type!==q.type) fail(`Missing or wrong answer type: ${id}.`);
    if(q.type==='noul') { if(!unit(a.noul)) fail(`Invalid probability: ${id}.`); continue; }
    const names=q.type==='choice'?Object.keys(q.criteria):q.criteria.map((_,i)=>String(i));
    if(!record(a.probabilities) || Object.keys(a.probabilities).length!==names.length || names.some(k=>!own(a.probabilities,k)||!unit(a.probabilities[k]))) fail(`Invalid distribution: ${id}.`);
    // Provider rounds individual probabilities; tolerate at most half a hundredth per option.
    if(Math.abs(Object.values(a.probabilities).reduce((x,y)=>x+y,0)-1)>names.length*0.005+1e-8) fail(`Distribution does not sum to one: ${id}.`);
    if(!unit(a.confidence)) fail(`Invalid confidence: ${id}.`);
    if(q.type==='choice' && !names.includes(a.choice)) fail(`Out-of-set choice: ${id}.`);
    if(q.type==='score' && !(typeof a.score==='number' && Number.isFinite(a.score) && a.score>=0 && a.score<=names.length-1)) fail(`Invalid score: ${id}.`);
  }
  return data;
}
export function gate(answer, threshold) {
  if(!Number.isFinite(threshold)||threshold<0||threshold>1) fail('Threshold must be between zero and one.');
  return answer.type==='noul' ? (answer.noul>=threshold?'positive':answer.noul<=1-threshold?'negative':'review') : (answer.confidence>=threshold?'accept':'review');
}
// Shared loopback contract for reader-supplied local endpoints (Kev in the lab, the local vision server in the image lab).
// One message per constraint, so a rejected endpoint names the rule it broke instead of failing silently.
// Any loopback port is accepted deliberately: the field exists so a reader can point at their own local server.
export function loopbackEndpoint(value) {
  let u; try { u=new URL(value); } catch { fail('Enter a loopback URL, for example http://127.0.0.1:8009/v1/systemone.'); }
  if(!['http:','https:'].includes(u.protocol)) fail(`Endpoint protocol must be http or https, got ${u.protocol}`);
  if(!['localhost','127.0.0.1','[::1]'].includes(u.hostname)) fail(`Endpoint host must be exactly localhost, 127.0.0.1 or [::1], got ${u.hostname}. Any port on those hosts is allowed.`);
  if(u.username||u.password) fail('Endpoint must not embed credentials.');
  if(u.search) fail('Endpoint must not carry a query string.');
  if(u.hash) fail('Endpoint must not carry a fragment.');
  if(u.pathname!=='/v1/systemone') fail(`Endpoint path must be exactly /v1/systemone, got ${u.pathname}.`);
  return u.href;
}
// INV-FETCH-HYGIENE: the transport options every hosted-provider call must carry. A 307/308
// makes the browser REPLAY the POST body — reader questions, state, an uploaded image — at the
// redirect target; ambient credentials or a Referer would leak the key or the page URL along
// with it. Spread this into every provider fetch init (post() below, the playground's hosted
// Jev call, image-lab's hosted TypeSafe call) so the policy has one source.
export const PROVIDER_FETCH_OPTIONS = { redirect: 'error', credentials: 'omit', referrerPolicy: 'no-referrer' };
async function post(url, headers, body, signal) {
  let response;
  try { response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify(body),signal,...PROVIDER_FETCH_OPTIONS}); }
  catch(e) { if(signal?.aborted) throw new Error('Request cancelled or timed out. No automatic retry.'); throw new Error('Connection failed or browser CORS refused. Jev rejected the Pages origin in our preflight check. Serve locally and select the optional relay for hosted APIs; for Kev check the server and local-network permission. No automatic retry.'); }
  if(!response.ok) throw new Error(`Provider returned HTTP ${response.status}. Check access, model name or rate limits. No automatic retry.`);
  return response.json();
}
function providerURL(provider, settings, direct) {
  if(settings.transport!=='relay') return direct;
  if(typeof location==='undefined'||!['127.0.0.1','localhost','[::1]'].includes(location.hostname)) fail('Relay is available only when serving this site locally.');
  return new URL('/api/'+provider,location.origin).href;
}
/** True when this script runs on a loopback-served page, or outside a browser entirely (node
 * harnesses, where `location` is undefined). EVERY test seam is honoured only here: the
 * __MOCK_DECIDE__ dispatch seam below, and the __LAYA_URLS/__KEV_URLS asset overrides in the
 * engine drivers (learnings-dkd, tm-test-seam-ungated). Shipped live, a seam would let any
 * injected script on the published origin short-circuit engine dispatch or point the 375-680 MB
 * artifact downloads at arbitrary endpoints and stall the tab; on any other host they are dead
 * code. Same gate as the relay transport above. */
export function onLoopback() {
  return typeof location === 'undefined' || ['127.0.0.1','localhost','[::1]'].includes(location.hostname);
}

// The __MOCK_DECIDE__ seam exists for the local browser harness (tests/browser.mjs serves over
// loopback). It short-circuits engine dispatch, bypassing validateAnswers/validateSpec, so it
// lives only where the harness can run: see onLoopback.
export function mockDecide() {
  if (!onLoopback()) return null;
  return globalThis.__MOCK_DECIDE__ ?? null;
}
export const PROVIDER_TIMEOUT_MS = 45000;
export async function decide(spec, settings, signal=AbortSignal.timeout(PROVIDER_TIMEOUT_MS)) {
  const url=settings.provider==='jev'?providerURL('jev',settings,'https://api.typesafe.ai/v1/systemone'):loopbackEndpoint(settings.endpoint);
  if(settings.provider==='jev'&&!settings.key?.trim()) fail('Enter a Jev API key first.');
  const start=performance.now();
  const data=validateAnswers(await post(url,settings.provider==='jev'?{Authorization:`Bearer ${settings.key}`}:{},requestFor(spec,settings.model),signal),spec.questions);
  return {data,elapsed:performance.now()-start,source:settings.provider==='jev'?'Jev API':'Kev local server',requestedModel:settings.model};
}
export const GENERATOR_PROMPT = `You design small educational decision-model experiments. Return ONLY a JSON object with exactly title (string), description (string), state (string), questions (object keyed by lowercase IDs). Each question has type, instructions (string), and optionally criteria. Types: noul = probability of yes, criteria optional {true:string,false:string}; choice = select among criteria object mapping 2–255 labels to string descriptions; score = expected index across criteria array of 2–10 ordered string levels. At least one question. No other fields. These are System One decisions: text context and bounded choices, not prose generation or chain-of-thought. Keep questions independent: they see the same state, not each other's answers. Include an abstain option where useful. Code, not the model, must do arithmetic, enforce permissions and perform actions. Never request keys, fetch URLs or emit HTML/JavaScript. Produce a meaningful state and rubric for the user's idea. The renderer is fixed: input editor, distributions, confidence and threshold. You cannot invent extra UI or code.`;
export async function generate(idea, settings, signal=AbortSignal.timeout(PROVIDER_TIMEOUT_MS)) {
  text(idea,'Idea'); text(settings.key,'Generator API key'); text(settings.model,'Generator model');
  const claude=settings.provider==='claude';
  const data=await post(providerURL(claude?'claude':'openai',settings,claude?'https://api.anthropic.com/v1/messages':'https://api.openai.com/v1/chat/completions'),claude?{'x-api-key':settings.key,'anthropic-version':'2023-06-01','anthropic-dangerous-direct-browser-access':'true'}:{Authorization:`Bearer ${settings.key}`},claude?{model:settings.model,max_tokens:4096,system:GENERATOR_PROMPT,messages:[{role:'user',content:idea}]}:{model:settings.model,messages:[{role:'system',content:GENERATOR_PROMPT},{role:'user',content:idea}],response_format:{type:'json_object'}},signal);
  const content=claude?data.content?.filter(x=>x.type==='text').map(x=>x.text).join('\n'):data.choices?.[0]?.message?.content;
  if(typeof content!=='string') fail('Generator returned no textual specification.');
  let spec;try { spec=JSON.parse(content.replace(/^```(?:json)?\s*\n?/, '').replace(/\n?```\s*$/, '')); } catch { fail('Generator did not return valid JSON. Nothing was executed.'); }
  return validateSpec(spec);
}
export function illustrative(spec, strength=.78) {
  validateSpec(spec);
  const answers=Object.fromEntries(Object.entries(spec.questions).map(([id,q])=>{
    if(q.type==='noul') return [id,{type:'noul',noul:strength}];
    const names=q.type==='choice'?Object.keys(q.criteria):q.criteria.map((_,i)=>String(i));
    const probabilities=Object.fromEntries(names.map((k,i)=>[k,i===0?strength:(1-strength)/(names.length-1)]));
    const confidence=(strength-1/names.length)/(1-1/names.length);
    return [id,q.type==='choice'?{type:'choice',choice:names[0],probabilities,confidence:Math.max(0,confidence)}:{type:'score',score:Object.values(probabilities).reduce((sum,p,i)=>sum+p*i,0),probabilities,confidence:Math.max(0,confidence)}];
  }));
  return {data:{model:'synthetic-distribution-not-a-model',answers},elapsed:0,source:'ILLUSTRATIVE — fixed distribution, no AI',requestedModel:'none'};
}
export function trialMetrics(answer, expected, threshold) {
  const p=answer.probabilities;
  if(answer.type!=='choice'||!own(p,expected)) fail('Expected label must be one of the Choice options.');
  return {correct:answer.choice===expected,expectedProbability:p[expected],brier:Object.entries(p).reduce((sum,[k,v])=>sum+(v-(k===expected?1:0))**2,0),gate:gate(answer,threshold)};
}
