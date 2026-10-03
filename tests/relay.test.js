import {test} from 'node:test';
import assert from 'node:assert/strict';
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
