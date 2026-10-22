import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync,writeFileSync,symlinkSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {serve} from '../scripts/serve.mjs';
test('loopback relay requires exact host, origin and fixed route; forwards only provider headers',async()=>{
 const original=globalThis.fetch,calls=[];const local=await serve();
 globalThis.fetch=async(url,options)=>{calls.push({url,options});return new Response('{"fixture":true}',{status:200,headers:{'Content-Type':'application/json'}});};
 try{
  for(const [path,headers] of [['api/jev',{}],['api/unknown',{'Origin':local.url.slice(0,-1)}],['api/jev',{'Origin':'https://evil.invalid'}]]){
   const r=await original(local.url+path,{method:'POST',headers:{'Content-Type':'application/json',...headers},body:'{}'});assert.equal(r.status,403);
  }
  assert.equal(calls.length,0);
  const r=await original(local.url+'api/jev',{method:'POST',headers:{'Origin':local.url.slice(0,-1),'Content-Type':'application/json','Authorization':'Bearer fake-test-key','Cookie':'must-not-forward','X-Other':'must-not-forward'},body:'{"state":"synthetic"}'});
  assert.equal(r.status,200);assert.equal(calls[0].url,'https://api.typesafe.ai/v1/systemone');assert.deepEqual(Object.keys(calls[0].options.headers),['Content-Type','Authorization']);assert.equal(calls[0].options.redirect,'error');
  const huge=await original(local.url+'api/jev',{method:'POST',headers:{'Origin':local.url.slice(0,-1),'Content-Type':'application/json'},body:'{"x":"'+'a'.repeat(1050000)+'"}'});
  assert.equal(huge.status,413);assert.equal(calls.length,1);
 }finally{globalThis.fetch=original;await new Promise(r=>local.server.close(r));}
});

// learnings-4jr: the relay used to write the response head and then await
// upstream.text(), so a provider that died mid-body threw ERR_HTTP_HEADERS_SENT
// from the catch and killed the process that also serves the static site.
test('a failing upstream body is a 502 and never takes the server down',async()=>{
 const original=globalThis.fetch,local=await serve();
 try{
  globalThis.fetch=async()=>new Response(new ReadableStream({start(c){c.enqueue(new TextEncoder().encode('{"partial":'));c.error(new Error('connection reset mid-body'));}}),{status:200,headers:{'Content-Type':'application/json'}});
  const failed=await original(local.url+'api/jev',{method:'POST',headers:{Origin:local.url.slice(0,-1),'Content-Type':'application/json'},body:'{}'});
  assert.equal(failed.status,502);
  assert.deepEqual(await failed.json(),{error:'Upstream request failed'});
  // Same listener, same process: the static site is still served.
  const page=await original(local.url+'decision-models/');
  assert.equal(page.status,200);
  assert.match(await page.text(),/<html/i);
 }finally{globalThis.fetch=original;await new Promise(r=>local.server.close(r));}
});

// learnings-6rq: the request body was capped but the upstream response was read
// whole. Both a declared oversize and a chunked oversize body are refused.
test('an oversized upstream response is refused instead of buffered without a bound',async()=>{
 const original=globalThis.fetch,local=await serve();
 const post=()=>original(local.url+'api/jev',{method:'POST',headers:{Origin:local.url.slice(0,-1),'Content-Type':'application/json'},body:'{}'});
 try{
  globalThis.fetch=async()=>new Response('{}',{status:200,headers:{'Content-Type':'application/json','Content-Length':String(9*1024*1024)}});
  const declared=await post();
  assert.equal(declared.status,502);
  assert.deepEqual(await declared.json(),{error:'Upstream response too large'});
  globalThis.fetch=async()=>new Response('{"pad":"'+'a'.repeat(8*1024*1024)+'"}',{status:200,headers:{'Content-Type':'application/json'}});
  const streamed=await post();
  assert.equal(streamed.status,502);
  assert.deepEqual(await streamed.json(),{error:'Upstream response too large'});
  assert.equal((await original(local.url+'decision-models/')).status,200);
 }finally{globalThis.fetch=original;await new Promise(r=>local.server.close(r));}
});

// learnings-9qz: containment was lexical only, so a symlink under the root was
// followed by readFile and served bytes from outside it.
test('a symlink under the served root cannot read outside it',async()=>{
 const outside=mkdtempSync(join(tmpdir(),'serve-outside-')),root=mkdtempSync(join(tmpdir(),'serve-root-'));
 writeFileSync(join(root,'index.html'),'<h1>inside</h1>');
 writeFileSync(join(outside,'secret.txt'),'SECRET-BYTES');
 symlinkSync(join(outside,'secret.txt'),join(root,'escape.txt'));
 symlinkSync(outside,join(root,'escape-dir'));
 const local=await serve({root});
 try{
  assert.equal((await fetch(local.url)).status,200);
  const file=await fetch(local.url+'escape.txt');
  assert.equal(file.status,404);
  assert.doesNotMatch(await file.text(),/SECRET-BYTES/);
  const dir=await fetch(local.url+'escape-dir/secret.txt');
  assert.equal(dir.status,404);
  assert.doesNotMatch(await dir.text(),/SECRET-BYTES/);
  // A hostile traversal is still refused, and the resolved root keeps working.
  assert.equal((await fetch(local.url+'..%2f..%2fetc%2fhostname')).status,404);
 }finally{await new Promise(r=>local.server.close(r));rmSync(outside,{recursive:true,force:true});rmSync(root,{recursive:true,force:true});}
});
