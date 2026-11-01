import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {validateSpec,validateAnswers,gate,loopbackEndpoint,illustrative,trialMetrics,requestFor,decide,generate,mockDecide,PROVIDER_FETCH_OPTIONS,readJSONBounded,RESPONSE_MAX_BYTES} from '../site/decision-models/core.js';
import {examples} from '../site/decision-models/examples.js';
for(const [name,s] of Object.entries(examples))test(`schema and synthetic roundtrip: ${name}`,()=>{assert.deepEqual(validateSpec(s),s);validateAnswers(illustrative(s).data,s.questions);});
test('generated executable fields and prototype IDs are refused',()=>{assert.throws(()=>validateSpec({...examples.tools,script:'alert(1)'}));assert.throws(()=>validateSpec({...examples.tools,questions:JSON.parse('{"__proto__":{"type":"noul","instructions":"test"}}')}));});
test('shape, null, empty instructions and out-of-contract levels rejected',()=>{for(const s of [null,{}, {...examples.tools,state:''},{...examples.tools,questions:{a:{type:'chat',instructions:'text'}}},{...examples.score,questions:{a:{type:'score',instructions:'x',criteria:['a']}}}])assert.throws(()=>validateSpec(s));});
test('Noul has two-sided abstention, Choice gates on confidence not pmax',()=>{assert.equal(gate({type:'noul',noul:.05},.8),'negative');assert.equal(gate({type:'noul',noul:.5},.8),'review');assert.equal(gate({type:'noul',noul:.8},.8),'positive');assert.equal(gate({type:'choice',confidence:.7},.8),'review');assert.throws(()=>gate({},NaN));});
test('loopback boundary refuses remote hosts, redirects-in-query and URL credentials',()=>{assert.equal(loopbackEndpoint('http://127.0.0.1:8009/v1/systemone'),'http://127.0.0.1:8009/v1/systemone');for(const v of ['https://evil.test/v1/systemone','http://localhost.evil.test/v1/systemone','http://user:secret@localhost/v1/systemone','javascript:alert(1)','http://localhost/v1/systemone?to=https://evil.test','http://localhost/other'])assert.throws(()=>loopbackEndpoint(v));});
test('loopback accepted-set is pinned: any loopback port and http or https, route fixed, each refusal names its constraint',()=>{
 // Any loopback port is accepted deliberately: the endpoint is reader-supplied so a reader can point at their
 // own local server (Ollama 11434, llama.cpp 8080, LM Studio 1234). https on loopback is accepted too - a hostile
 // local TLS server fails on the certificate before any payload is posted. Do not tighten without deciding that.
 for(const [v,href] of [['http://127.0.0.1:8009/v1/systemone','http://127.0.0.1:8009/v1/systemone'],['http://localhost:11434/v1/systemone','http://localhost:11434/v1/systemone'],['http://[::1]:1234/v1/systemone','http://[::1]:1234/v1/systemone'],['https://127.0.0.1:8009/v1/systemone','https://127.0.0.1:8009/v1/systemone']])assert.equal(loopbackEndpoint(v),href);
 const refused=[['http://127.0.0.1:8009/other',/path must be exactly \/v1\/systemone/],['http://127.0.0.1:8009/v1/systemone?to=https://evil.test',/must not carry a query string/],['http://127.0.0.1:8009/v1/systemone#frag',/must not carry a fragment/],['http://user:secret@localhost/v1/systemone',/must not embed credentials/],['http://localhost.evil.test/v1/systemone',/host must be exactly localhost/],['ftp://127.0.0.1/v1/systemone',/protocol must be http or https/],['nonsense',/Enter a loopback URL/]];
 for(const [v,reason] of refused)assert.throws(()=>loopbackEndpoint(v),reason,`refusal of ${v} must name the constraint that failed`);
});
test('response is fail-closed for missing answers, unknown options, non-finite and bad distributions',()=>{const s=examples.tools,d=illustrative(s).data;for(const mutate of [d=>delete d.answers.action,d=>d.answers.action.choice='run_shell',d=>d.answers.action.confidence=NaN,d=>d.answers.action.probabilities.list_tabs=9,d=>d.answers.action.probabilities={list_tabs:1}]){const c=structuredClone(d);mutate(c);assert.throws(()=>validateAnswers(c,s.questions));}});
test('multiclass Brier formula and expected label validation',()=>{const a={type:'choice',choice:'a',confidence:.8,probabilities:{a:.8,b:.2}};const m=trialMetrics(a,'b',.9);assert.equal(m.correct,false);assert.ok(Math.abs(m.brier-1.28)<1e-10);assert.equal(m.gate,'review');assert.throws(()=>trialMetrics(a,'c',.8));});
test('illustration ignores text and discloses synthetic provenance',()=>{const a=illustrative(examples.tools),b=illustrative({...examples.tools,state:'Anything entirely different'});assert.deepEqual(a,b);assert.match(a.source,/ILLUSTRATIVE/);assert.match(a.data.model,/not-a-model/);});
test('request excludes keys and generator fields',()=>{const r=requestFor(examples.tools,'jev-1.13.0');assert.deepEqual(Object.keys(r),['model','state','questions']);});
test('real decision adapters use correct host, key boundary and validation (mock transport)',async()=>{
 const original=globalThis.fetch,calls=[];globalThis.fetch=async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>illustrative(examples.tools).data};};
 try{await decide(examples.tools,{provider:'jev',key:'synthetic-test-only',model:'jev-1.13.0'});await decide(examples.tools,{provider:'kev',key:'must-not-send',endpoint:'http://localhost:8009/v1/systemone',model:'kev-latest'});assert.equal(calls[0].url,'https://api.typesafe.ai/v1/systemone');assert.equal(calls[0].options.headers.Authorization,'Bearer synthetic-test-only');assert.equal(calls[1].options.headers.Authorization,undefined);assert.equal(calls[0].options.redirect,'error');assert.equal(calls[0].options.credentials,'omit');
 // learnings-pv9s: the Kev loopback transport is core.js's shared postJSON too, so it must carry
 // the same hardening as the hosted call - pin it here so a sibling refactor cannot drop it.
 assert.equal(calls[1].url,'http://localhost:8009/v1/systemone');
 for(const [option,value] of Object.entries(PROVIDER_FETCH_OPTIONS))assert.equal(calls[1].options[option],value,`the kev loopback call carries ${option}: ${value}`);}finally{globalThis.fetch=original;}
});
test('OpenAI and Claude generation validate JSON and reject code (mock transport)',async()=>{const original=globalThis.fetch,calls=[];globalThis.fetch=async(url,options)=>{calls.push({url,options});return {ok:true,json:async()=>url.includes('anthropic')?{content:[{type:'text',text:JSON.stringify(examples.tools)}]}:{choices:[{message:{content:JSON.stringify(examples.tools)}}]}};};try{for(const provider of ['openai','claude'])assert.deepEqual(await generate('test',{provider,key:'synthetic-test-only',model:'test'}),examples.tools);assert.equal(calls[1].options.headers['anthropic-dangerous-direct-browser-access'],'true');globalThis.fetch=async()=>({ok:true,json:async()=>({choices:[{message:{content:'<script>alert(1)</script>'}}]})});await assert.rejects(()=>generate('test',{provider:'openai',key:'synthetic',model:'test'}),/valid JSON/);}finally{globalThis.fetch=original;}});
test('__MOCK_DECIDE__ seam is honoured on loopback and dead off-loopback (learnings-ebg)',()=>{
 const seam=()=>({mocked:true});globalThis.__MOCK_DECIDE__=seam;const saved=Object.getOwnPropertyDescriptor(globalThis,'location');
 try{
  // No location (node harness): the harness owns the realm, so the seam works.
  assert.equal(mockDecide(),seam);
  for(const host of ['127.0.0.1','localhost','[::1]']){
   Object.defineProperty(globalThis,'location',{configurable:true,value:{hostname:host}});
   assert.equal(mockDecide(),seam,`${host}: the local harness keeps the seam`);
  }
  for(const host of ['paul.kinlan.me','localhost.evil.test','127.0.0.1.evil.test']){
   Object.defineProperty(globalThis,'location',{configurable:true,value:{hostname:host}});
   assert.equal(mockDecide(),null,`${host}: a deployed page must not honour the seam`);
  }
 }finally{
  delete globalThis.__MOCK_DECIDE__;
  if(saved)Object.defineProperty(globalThis,'location',saved);else delete globalThis.location;
 }
 assert.equal(mockDecide(),null,'no seam installed -> no mock');
});

// learnings-is49 (tm-provider-response-unbounded-read): provider/loopback bodies are read
// through a bounded stream BEFORE parsing. The bound is pinned from four sides: an ordinary
// body parses (with a multibyte character split across chunks, proving the merged-buffer
// decode), an over-budget body is refused mid-stream with the reader CANCELLED instead of
// drained, a declared Content-Length over budget is refused before a byte is read, and a
// source pin keeps post() from silently regressing to response.json().
const streamOf=(chunks)=>new ReadableStream({start(c){for(const b of chunks)c.enqueue(b);c.close();}});
test('readJSONBounded parses an ordinary body with a multibyte char split across chunks',async()=>{
 const payload=JSON.stringify({answers:{tool:{choice:'café ✓'}}});
 const bytes=new TextEncoder().encode(payload);
 const at=bytes.indexOf(0xC3)+1; // inside the two-byte é, so the chunks alone are invalid UTF-8
 const parsed=await readJSONBounded(new Response(streamOf([bytes.slice(0,at),bytes.slice(at)])));
 assert.deepEqual(parsed,JSON.parse(payload));
});
test('readJSONBounded refuses an over-budget body and cancels the stream instead of draining it',async()=>{
 let produced=0;
 const stream=new ReadableStream({pull(c){if(produced>=64*1024){c.close();return;}produced+=1024;c.enqueue(new Uint8Array(1024).fill(97));}});
 await assert.rejects(()=>readJSONBounded(new Response(stream),4096),/exceeded the 4096-byte budget after \d+ bytes/);
 assert.ok(produced<=8192,`the reader cancelled early instead of draining the sender: produced ${produced} of 65536`);
});
test('readJSONBounded refuses a declared Content-Length over budget without reading a byte',async()=>{
 // A constructed Response cannot carry this case: undici drops a hand-set content-length on a
 // stream body (it cannot know the length), so the declared-length branch is exercised on a
 // Response-shaped object with a real Headers and a POISON body — Node primes a ReadableStream's
 // pull() even with no consumer, so a pull-tracked stream cannot prove the body went unread;
 // a getReader() that fails the test can.
 const shaped={headers:new Headers({'content-length':String(2*RESPONSE_MAX_BYTES)}),
   body:{getReader:()=>assert.fail('the body must not be read when the declared length is over budget')},
   json:()=>assert.fail('json() fallback must not run')};
 await assert.rejects(()=>readJSONBounded(shaped,RESPONSE_MAX_BYTES),/declares \d+ bytes, over the \d+-byte budget/);
});
test('post() reads through the bound, and the shipped bound is 8 MiB',()=>{
 assert.equal(RESPONSE_MAX_BYTES,8*1024*1024);
 const src=readFileSync(new URL('../site/decision-models/core.js',import.meta.url),'utf8');
 assert.match(src,/return readJSONBounded\(response\);/,'post() must hand the response to the bounded reader');
});
