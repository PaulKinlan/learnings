// Content-identifies the tree a gate actually measured. A receipt named only for
// HEAD cannot tie a PASS on an uncommitted worktree to what it measured
// (learnings-j38): the same HEAD can produce many different measured states, and
// a reviewer clearance must be tied to the content, the way fleet-check pins
// HEAD^{tree}. The measured tree is computed through a throwaway index so the
// run's own index and working tree are never touched: read-tree seeds the
// throwaway from HEAD, add -A mirrors the working tree (respecting .gitignore),
// and write-tree content-addresses the result — identical content always agrees.
import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

export function measureTree(repoRoot){
  const run=(args,env)=>execFileSync('git',args,{cwd:repoRoot,encoding:'utf8',env:{...process.env,...env}}).trim();
  const head=run(['rev-parse','HEAD']);
  const headTree=run(['rev-parse','HEAD^{tree}']);
  const scratch=mkdtempSync(join(tmpdir(),'gate-index-'));
  let tree;
  try{
    const indexEnv={GIT_INDEX_FILE:join(scratch,'index')};
    run(['read-tree','HEAD'],indexEnv);
    run(['add','-A'],indexEnv);
    tree=run(['write-tree'],indexEnv);
  }finally{rmSync(scratch,{recursive:true,force:true});}
  return {head,commit:head.slice(0,12),headTree,tree,dirty:tree!==headTree};
}
