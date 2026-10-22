import http from 'node:http';
import {readFile,realpath,stat} from 'node:fs/promises';
import {resolve,extname,sep,join} from 'node:path';
import {fileURLToPath} from 'node:url';

// Bounds for the developer loopback relay. Both sides are capped: an unbounded
// upstream read is the same defect class as the unbounded download this repo
// already fixed (learnings-6rq), and reading the whole body BEFORE writing the
// head is what keeps a dying provider from taking the reader's lab down
// (learnings-4jr).
const MAX_REQUEST_BYTES=1048576;
const MAX_UPSTREAM_BYTES=8*1024*1024;

const inside=(root,target)=>target===root.replace(/[\\/]$/,'')||target.startsWith(root.endsWith(sep)?root:root+sep);

// Read an upstream body under a hard byte cap. The body is consumed in full and
// returned as text so the caller can only write response headers once it has
// something real to send: a stream that errors or overflows rejects here, before
// any header reaches the socket.
async function boundedText(body,limit){
 if(!body)return '';
 if(typeof body.getReader!=='function'){
  const text=await body.text();
  if(Buffer.byteLength(text)>limit)throw new RangeError('Upstream response too large');
  return text;
 }
 const reader=body.getReader(),chunks=[];let total=0;
 try{
  for(;;){
   const {done,value}=await reader.read();
   if(done)break;
   total+=value?.byteLength??0;
   if(total>limit)throw new RangeError('Upstream response too large');
   chunks.push(value);
  }
 }finally{reader.cancel().catch(()=>{});}
 return Buffer.concat(chunks.map(chunk=>Buffer.from(chunk))).toString('utf8');
}

export async function serve({root:servedRoot}={}){
 const root=servedRoot??fileURLToPath(new URL('../site/',import.meta.url));
 // Containment is checked against the resolved root and the resolved file, so a
 // symlink under the root cannot read files outside it (learnings-9qz).
 const rootReal=await realpath(root);
 const upstreams={'/api/jev':'https://api.typesafe.ai/v1/systemone','/api/openai':'https://api.openai.com/v1/chat/completions','/api/claude':'https://api.anthropic.com/v1/messages'};
 const server=http.createServer(async(req,res)=>{try{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const host=`127.0.0.1:${server.address().port}`;
  if(req.headers.host!==host){res.writeHead(403);res.end('Use the printed loopback URL.');return;}
  if(pathname.startsWith('/api/')){
   if(req.method!=='POST'||req.headers.origin!==`http://${host}`||!req.headers['content-type']?.startsWith('application/json')||!Object.hasOwn(upstreams,pathname)){res.writeHead(403);res.end('Relay requires same-origin JSON POST to a fixed provider route.');return;}
   const chunks=[];let total=0;for await(const chunk of req){total+=chunk.length;if(total>MAX_REQUEST_BYTES){res.writeHead(413);res.end('Payload too large');return;}chunks.push(chunk);}
   const body=Buffer.concat(chunks).toString('utf8');JSON.parse(body);
   const headers={'Content-Type':'application/json'};
   if(pathname==='/api/claude'){headers['x-api-key']=req.headers['x-api-key']??'';headers['anthropic-version']='2023-06-01';}else headers.Authorization=req.headers.authorization??'';
   // Buffer and verify the whole upstream response first; only then write the head.
   try{
    const upstream=await fetch(upstreams[pathname],{method:'POST',headers,body,redirect:'error',signal:AbortSignal.timeout(45000)});
    const declared=Number(upstream.headers.get('content-length'));
    if(Number.isFinite(declared)&&declared>MAX_UPSTREAM_BYTES)throw new RangeError('Upstream response too large');
    const text=await boundedText(upstream.body,MAX_UPSTREAM_BYTES);
    res.writeHead(upstream.status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(text);
   }catch(error){
    if(res.headersSent){res.destroy();return;}
    res.writeHead(502,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(JSON.stringify({error:error instanceof RangeError?'Upstream response too large':'Upstream request failed'}));
   }
   return;
  }
  let file=resolve(root,'.'+pathname);if(!inside(root,file))throw new Error('outside site');
  let real=await realpath(file);if((await stat(real)).isDirectory())real=await realpath(join(real,'index.html'));
  if(!inside(rootReal,real))throw new Error('outside site');
  const bytes=await readFile(real);res.writeHead(200,{'Content-Type':({'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.wasm':'application/wasm','.md':'text/markdown; charset=utf-8'})[extname(real)]??'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(bytes);}catch{if(res.headersSent){res.destroy();return;}res.writeHead(404);res.end('Not found');}});
 await new Promise(r=>server.listen(Number(process.env.PORT??0),'127.0.0.1',r));return {server,url:`http://127.0.0.1:${server.address().port}/`};
}
if(process.argv[1]===fileURLToPath(import.meta.url)){const s=await serve();console.log(s.url);}
