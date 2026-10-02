import http from 'node:http';
import {readFile,stat} from 'node:fs/promises';
import {resolve,extname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
export async function serve(){
 const root=fileURLToPath(new URL('../site/',import.meta.url));
 const upstreams={'/api/jev':'https://api.typesafe.ai/v1/systemone','/api/openai':'https://api.openai.com/v1/chat/completions','/api/claude':'https://api.anthropic.com/v1/messages'};
 const server=http.createServer(async(req,res)=>{try{const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  const host=`127.0.0.1:${server.address().port}`;
  if(req.headers.host!==host){res.writeHead(403);res.end('Use the printed loopback URL.');return;}
  if(pathname.startsWith('/api/')){
   if(req.method!=='POST'||req.headers.origin!==`http://${host}`||!req.headers['content-type']?.startsWith('application/json')||!Object.hasOwn(upstreams,pathname)){res.writeHead(403);res.end('Relay requires same-origin JSON POST to a fixed provider route.');return;}
   const chunks=[];let total=0;for await(const chunk of req){total+=chunk.length;if(total>1048576){res.writeHead(413);res.end('Payload too large');return;}chunks.push(chunk);}
   const body=Buffer.concat(chunks).toString('utf8');JSON.parse(body);
   const headers={'Content-Type':'application/json'};
   if(pathname==='/api/claude'){headers['x-api-key']=req.headers['x-api-key']??'';headers['anthropic-version']='2023-06-01';}else headers.Authorization=req.headers.authorization??'';
   const upstream=await fetch(upstreams[pathname],{method:'POST',headers,body,redirect:'error',signal:AbortSignal.timeout(45000)});
   res.writeHead(upstream.status,{'Content-Type':'application/json','Cache-Control':'no-store'});res.end(await upstream.text());return;
  }
  let file=resolve(root,'.'+pathname);if(!file.startsWith(root.endsWith(sep)?root:root+sep)&&file!==resolve(root))throw new Error('outside site');if((await stat(file)).isDirectory())file=resolve(file,'index.html');const bytes=await readFile(file);res.writeHead(200,{'Content-Type':({'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json','.wasm':'application/wasm','.md':'text/markdown; charset=utf-8'})[extname(file)]??'application/octet-stream','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});res.end(bytes);}catch{res.writeHead(404);res.end('Not found');}});
 await new Promise(r=>server.listen(Number(process.env.PORT??0),'127.0.0.1',r));return {server,url:`http://127.0.0.1:${server.address().port}/`};
}
if(process.argv[1]===fileURLToPath(import.meta.url)){const s=await serve();console.log(s.url);}
