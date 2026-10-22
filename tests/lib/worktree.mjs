// Content-identifies the tree a gate actually measured. A receipt named only for
// HEAD cannot tie a PASS on an uncommitted worktree to what it measured
// (learnings-j38): the same HEAD can produce many different measured states, and
// a reviewer clearance must be tied to the content, the way fleet-check pins
// HEAD^{tree}. The measured tree is computed through a throwaway index so the
// run's own index and working tree are never touched: read-tree seeds the
// throwaway from HEAD, add -A mirrors the working tree (respecting .gitignore),
// and write-tree content-addresses the result — identical content always agrees.
//
// Symlinks (learnings-2pr): add -A records only a link's path (mode 120000),
// but scripts/serve.mjs reads through readFile, which FOLLOWS links — so
// overwriting a link's target changed the served bytes while this id stood
// still. Every staged symlink entry is therefore replaced with the bytes a
// link-following reader would actually be served (a link to a directory
// contributes the files under its resolved target). A committed symlink thus
// reports dirty even when untouched: the literal HEAD tree does not describe
// the served bytes. The returned links manifest records what was resolved so a
// receipt can show it.
//
// Nesting (learnings-tuj): a fixed depth cap silently dropped everything below it,
// so a deep file behind a directory link changed served bytes without moving the
// id. The walk is iterative and uncapped instead — depth is bounded only by the
// filesystem — and a link cycle throws an explicit error rather than looping or
// truncating, so the gate either measures every served file or fails loudly.
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync,readlinkSync,readdirSync,realpathSync,readFileSync,statSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

// Deterministic stand-in for a link a reader cannot resolve (dangling target,
// unreadable file, non-regular target): retargeting or repairing it changes the id.
const unreadable=target=>Buffer.from(`gate-receipt: link target not readable (${target})`);

// Entries a link-following reader would be served for one staged symlink path.
function servedEntries(abs,path,target,real,kind){
  const entry=bytes=>({path,bytes});
  if(kind==='file'){
    try{return [entry(readFileSync(real))];}catch{return [entry(unreadable(target))];}
  }
  if(kind!=='dir')return [entry(unreadable(target))];
  const entries=[];
  // Iterative and uncapped (learnings-tuj): a fixed cap silently truncated deep
  // directories. `ancestors` holds the realpaths on this branch, so a link that
  // loops back is an explicit error instead of an endless walk.
  const stack=[{dir:real,rel:'',ancestors:new Set([real])}];
  while(stack.length){
    const {dir,rel,ancestors}=stack.pop();
    let dirents;
    try{dirents=readdirSync(dir,{withFileTypes:true});}
    catch(e){throw new Error(`gate-receipt: cannot read ${dir} reached through ${rel?`${path}/${rel}`:path} (${e.code??e.message})`);}
    dirents.sort((a,b)=>a.name<b.name?-1:a.name>b.name?1:0);
    for(const d of dirents){
      const childPath=rel?`${path}/${rel}/${d.name}`:`${path}/${d.name}`;
      const childRel=rel?`${rel}/${d.name}`:d.name;
      const childAbs=join(dir,d.name);
      if(d.isSymbolicLink()){
        const childTarget=readlinkSync(childAbs);
        let childReal,childKind='unreadable';
        try{childReal=realpathSync(childAbs);childKind=statSync(childReal).isDirectory()?'dir':'file';}catch{childReal=null;}
        if(childKind==='dir'){
          if(ancestors.has(childReal))throw new Error(`gate-receipt: symlink cycle at ${childPath} -> ${childTarget}`);
          stack.push({dir:childReal,rel:childRel,ancestors:new Set(ancestors).add(childReal)});
          continue;
        }
        try{entries.push({path:childPath,bytes:readFileSync(childReal??childAbs)});}
        catch{entries.push({path:childPath,bytes:unreadable(childTarget)});}
      }else if(d.isFile()){
        try{entries.push({path:childPath,bytes:readFileSync(childAbs)});}
        catch{entries.push({path:childPath,bytes:unreadable(childPath)});}
      }else if(d.isDirectory()){
        const childReal=realpathSync(childAbs);
        if(ancestors.has(childReal))throw new Error(`gate-receipt: directory cycle at ${childPath}`);
        stack.push({dir:childAbs,rel:childRel,ancestors:new Set(ancestors).add(childReal)});
      }
    }
  }
  return entries.length?entries:[entry(unreadable(target))]; // empty dir: no index entry would drop the link silently
}

export function measureTree(repoRoot){
  const run=(args,env,opts)=>execFileSync('git',args,{cwd:repoRoot,encoding:'utf8',env:{...process.env,...env},...opts}).trim();
  const head=run(['rev-parse','HEAD']);
  const headTree=run(['rev-parse','HEAD^{tree}']);
  const scratch=mkdtempSync(join(tmpdir(),'gate-index-'));
  let tree;
  const links=[];
  try{
    const indexEnv={GIT_INDEX_FILE:join(scratch,'index')};
    run(['read-tree','HEAD'],indexEnv);
    run(['add','-A'],indexEnv);
    // ls-files against the throwaway index lists exactly what add -A staged
    // (tracked + untracked, .gitignore respected), so symlink discovery stays
    // in step with what the tree records — no physical walk, no ignored dirs.
    for(const line of run(['ls-files','-s','-z'],indexEnv).split('\0')){
      if(!line)continue;
      const [meta,path]=line.split('\t');
      const mode=meta.slice(0,meta.indexOf(' '));
      if(mode!=='120000')continue;
      const abs=join(repoRoot,path);
      const target=readlinkSync(abs);
      let real=null,kind='unreadable';
      try{real=realpathSync(abs);kind=statSync(real).isDirectory()?'dir':'file';}catch{}
      links.push({path,target,kind});
      run(['update-index','--force-remove','--',path],indexEnv);
      for(const e of servedEntries(abs,path,target,real,kind)){
        const blob=run(['hash-object','-w','--stdin'],indexEnv,{input:e.bytes});
        run(['update-index','--add','--cacheinfo','100644',blob,e.path],indexEnv);
      }
    }
    tree=run(['write-tree'],indexEnv);
  }finally{rmSync(scratch,{recursive:true,force:true});}
  links.sort((a,b)=>a.path<b.path?-1:1);
  return {head,commit:head.slice(0,12),headTree,tree,dirty:tree!==headTree,links};
}
