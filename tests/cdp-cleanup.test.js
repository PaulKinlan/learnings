import assert from 'node:assert/strict';
import {execFileSync, spawn} from 'node:child_process';
import {existsSync, mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {assertNoBrowserOrphans, launch, resolveBinary} from './lib/cdp.mjs';

const binary = resolveBinary();
const waitForExit = (child) => new Promise((resolve, reject) => {
  if (child.exitCode !== null || child.signalCode !== null) return resolve();
  const timer = setTimeout(() => reject(new Error(`runner ${child.pid} did not exit`)), 10000);
  child.once('exit', () => { clearTimeout(timer); resolve(); });
});
const killGroup = (child) => {
  if (child.pid) {
    try { process.kill(-child.pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
  }
};
const reapTestProfile = (profile) => {
  // Last-resort test teardown if an assertion fails before the runner's cleanup completes.
  for (const line of execFileSync('ps', ['-eo', 'pid=,args='], {encoding: 'utf8'}).split('\n')) {
    if (!line.includes(`--user-data-dir=${profile}`) || !/chrom(e|ium)/.test(line)) continue;
    const pid = Number(line.trim().match(/^\d+/)?.[0]);
    if (pid) {
      try { process.kill(-pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    }
  }
};

test('cdp close removes its browser process group and temporary profile', {skip: !binary}, async () => {
  const page = await launch();
  const {profile} = page;
  try {
    assert.ok(existsSync(profile));
  } finally {
    await page.close();
  }
  await assertNoBrowserOrphans(profile);
  assert.equal(existsSync(profile), false);
  console.log(`PASS  normal close: no browser processes or profile ${profile}`);
});

test('SIGTERM interrupts a runner and removes its browser and profile', {skip: !binary}, async () => {
  const runner = spawn(process.execPath, ['--input-type=module', '-e', `
    import {launch} from ${JSON.stringify(new URL('./lib/cdp.mjs', import.meta.url).href)};
    const page = await launch();
    console.log('PROFILE=' + page.profile);
    setInterval(() => {}, 1000);
  `], {stdio: ['ignore', 'pipe', 'pipe'], detached: true});
  let profile;
  try {
    profile = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('runner did not launch')), 25000);
      let output = '';
      runner.stdout.on('data', (chunk) => {
        output += chunk;
        const found = output.match(/PROFILE=(\/tmp\/voicebox-cdp-[^\s]+)/);
        if (found) { clearTimeout(timer); resolve(found[1]); }
      });
      runner.once('exit', (code) => { clearTimeout(timer); reject(new Error(`runner exited before launch: ${code}`)); });
      runner.once('error', reject);
    });
    assert.ok(existsSync(profile));
    runner.kill('SIGTERM');
    await waitForExit(runner);
    await assertNoBrowserOrphans(profile);
    assert.equal(existsSync(profile), false);
    console.log(`PASS  SIGTERM: no browser processes or profile ${profile}`);
  } finally {
    killGroup(runner);
    if (profile) {
      reapTestProfile(profile);
      rmSync(profile, {recursive: true, force: true});
    }
  }
});

test('orphan assertion names a deliberately unclosed browser', {skip: !binary}, async () => {
  const profile = mkdtempSync(join(tmpdir(), 'voicebox-cdp-'));
  const browser = spawn(binary, ['--headless=new', '--no-sandbox', '--disable-gpu', `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank'], {stdio: ['ignore', 'ignore', 'pipe'], detached: true});
  try {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('deliberate orphan did not launch')), 20000);
      browser.stderr.on('data', (chunk) => {
        if (String(chunk).includes('DevTools listening')) { clearTimeout(timer); resolve(); }
      });
      browser.once('error', reject);
      browser.once('exit', (code) => { clearTimeout(timer); reject(new Error(`browser exited before assertion: ${code}`)); });
    });
    await assert.rejects(assertNoBrowserOrphans(profile), (error) =>
      error.message.includes(`profile ${profile}`) && error.message.includes(String(browser.pid)) && error.message.includes(`--user-data-dir=${profile}`));
    console.log(`PASS  orphan assertion failed as intended: pid ${browser.pid}, profile ${profile}`);
    await assert.rejects(launch({profile}), (error) =>
      error.message.includes(`cannot launch with occupied profile ${profile}`) && error.message.includes(String(browser.pid)));
    console.log(`PASS  start-time guard refused occupied profile ${profile}, pid ${browser.pid}`);
  } finally {
    killGroup(browser);
    reapTestProfile(profile);
    await waitForExit(browser);
    rmSync(profile, {recursive: true, force: true});
  }
  await assertNoBrowserOrphans(profile);
  assert.equal(existsSync(profile), false);
});
