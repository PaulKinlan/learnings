import assert from 'node:assert/strict';
import {execFileSync, spawn} from 'node:child_process';
import {existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';
import {assertNoBrowserOrphans, isPageEvent, launch, resolveBinary} from './lib/cdp.mjs';

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
// Chrome processes running on a given profile. The browser is in its own detached process group,
// so the profile on its command line is the only handle a test has on it.
const chromePidsFor = (profile) => {
  const pids = [];
  for (const line of execFileSync('ps', ['-eo', 'pid=,args='], {encoding: 'utf8'}).split('\n')) {
    if (!line.includes(`--user-data-dir=${profile}`) || !/chrom(e|ium)/.test(line)) continue;
    const pid = Number(line.trim().match(/^\d+/)?.[0]);
    if (pid) pids.push(pid);
  }
  return pids;
};
const reapTestProfile = (profile) => {
  // Last-resort test teardown if an assertion fails before the runner's cleanup completes.
  for (const pid of chromePidsFor(profile)) {
    try { process.kill(-pid, 'SIGKILL'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
  }
};

/**
 * Tear down a runner whose browser profile may never have reached the test, and reap what the
 * runner could not.
 *
 * SIGTERM BEFORE SIGKILL: the runner installs cdp.mjs's signal handlers, whose exit path kills the
 * browser's process group and removes the profile — the only cleanup that works when the launch
 * bound trips and the profile is unknown here (chrome is its own detached group, so killGroup
 * cannot reach it). A SIGKILL-only teardown orphaned that group, ~1.2GB per leak, until the fleet
 * reaper's 45-minute browser limit (observed profile voicebox-cdp-1I0shL, learnings-yci0). A runner
 * wedged past waitForExit's bound is still SIGKILLed afterwards, and a PROFILE= line that arrives
 * late is reaped too. `state` accumulates the runner's stdout, because that is where the profile
 * comes from when the launch did report it.
 *
 * Idempotent: a runner that already exited is only killGroup'd, which is a no-op on a dead group.
 */
const stopRunner = async (runner, state) => {
  if (runner.exitCode === null && runner.signalCode === null) {
    try { runner.kill('SIGTERM'); } catch (error) { if (error.code !== 'ESRCH') throw error; }
    await waitForExit(runner).catch(() => {});
  }
  killGroup(runner);
  const profile = state.output.match(/PROFILE=(\/tmp\/voicebox-cdp-[^\s]+)/)?.[1];
  if (profile) {
    reapTestProfile(profile);
    rmSync(profile, {recursive: true, force: true});
  }
};

// learnings-19b8: page.on() is checked here rather than in a test of its own on purpose — it needs
// a live browser, and a second launch would add ~5s (up to ~20s under fleet load) to every full
// gate run for assertions that fit in a suite that already has one open.
test('cdp close removes its browser process group and temporary profile; page.on delivers this page\'s events', {skip: !binary}, async () => {
  const page = await launch();
  const {profile} = page;
  try {
    assert.ok(existsSync(profile));
    const seen = [];
    const unsubscribe = page.on('Runtime.consoleAPICalled', params => seen.push(params));
    await page.evaluate(() => { console.log('first'); });
    for (let i = 0; i < 40 && seen.length === 0; i++) await new Promise(r => setTimeout(r, 25));
    assert.equal(seen.length, 1, 'the handler receives the event this page produced');
    assert.equal(seen[0].type, 'log', 'and the params, not the envelope');
    assert.equal(seen[0].args[0].value, 'first');

    unsubscribe();
    await page.evaluate(() => { console.log('second'); });
    await new Promise(r => setTimeout(r, 120));
    assert.equal(seen.length, 1, 'no delivery after unsubscribe');

    // A handler that throws must not wedge the driver: a throw escaping the socket listener would
    // take reply routing (every pending send) with it, so it is reported as a warning instead.
    const warnings = [];
    const onWarning = warning => warnings.push(warning.message);
    process.on('warning', onWarning);
    let calls = 0;
    const off = page.on('Runtime.consoleAPICalled', () => { calls++; throw new Error('probe bug'); });
    try {
      await page.evaluate(() => { console.log('third'); });
      for (let i = 0; i < 40 && warnings.length === 0; i++) await new Promise(r => setTimeout(r, 25));
      assert.equal(calls, 1, 'the throwing handler ran once');
      assert.equal(warnings.length, 1, 'the handler failure is reported, not swallowed');
      assert.match(warnings[0], /Runtime\.consoleAPICalled.*probe bug/);
      assert.equal(await page.evaluate(() => 1 + 1), 2, 'the driver still answers');
    } finally {
      off();
      process.removeListener('warning', onWarning);
    }
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
  // Hoisted out of the promise so stdout keeps accumulating after the launch bound trips: a
  // late PROFILE= line is then still enough for the finally block to identify and reap the
  // runner's browser (learnings-yci0).
  const state = {output: ''};
  try {
    profile = await new Promise((resolve, reject) => {
      // 90s: launch() retries a failed startup once (learnings-gib), so a slow-start box can
      // legitimately take two 20s endpoint bounds before the runner reports its profile — and
      // under fleet load (observed at load average ~10) even that exceeded the old 45s bound,
      // failing gates for changes that touch nothing here (learnings-p564). A runner that is
      // truly hung still trips this bound; the suite's own gate bound is far larger.
      const timer = setTimeout(() => reject(new Error('runner did not launch')), 90000);
      runner.stdout.on('data', (chunk) => {
        state.output += chunk;
        const found = state.output.match(/PROFILE=(\/tmp\/voicebox-cdp-[^\s]+)/);
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
    // The bound-trip cleanup, shared with the launch-bound test below so it is asserted rather
    // than only executed (learnings-bdli); its SIGTERM-before-SIGKILL ordering and the
    // late-PROFILE reap are documented on stopRunner.
    await stopRunner(runner, state);
  }
});

// learnings-bdli: the teardown above was the one cleanup in this file with no assertion behind it.
// Its happy-path leak check sits in the try, after the profile resolves, so the path that matters —
// the launch bound tripping while chrome is already up and the profile never reached the test — ran
// with nothing watching it, and a regression to a SIGKILL-only teardown still passed the suite.
// This test reproduces that window on purpose and asserts what the teardown left behind, with the
// one runner it spawns itself.
test('a launch bound that trips with chrome already up leaves no browser behind (learnings-bdli)', {skip: !binary}, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'cdp-bound-'));
  const marker = join(dir, 'profile');
  // The runner records its profile in a FILE rather than on stdout, because that is exactly the
  // bound-trip case: the browser is up and the teardown has no PROFILE= line to identify it by.
  // The file is this test's own bookkeeping, so the leak assertion has a profile to look for.
  const runner = spawn(process.execPath, ['--input-type=module', '-e', `
    import {writeFileSync} from 'node:fs';
    import {launch} from ${JSON.stringify(new URL('./lib/cdp.mjs', import.meta.url).href)};
    const page = await launch();
    writeFileSync(${JSON.stringify(marker)}, page.profile);
    setInterval(() => {}, 1000);
  `], {stdio: ['ignore', 'pipe', 'pipe'], detached: true});
  const state = {output: ''};
  runner.stdout.on('data', (chunk) => { state.output += chunk; });
  let profile = null;
  try {
    // Same 90s launch bound as the SIGTERM test above (learnings-gib retries a failed startup
    // once; learnings-p564 measured a slow box past the old 45s).
    profile = await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('runner did not launch')), 90000);
      const tick = setInterval(() => {
        if (!existsSync(marker)) return;
        clearInterval(tick);
        clearTimeout(timer);
        resolve(readFileSync(marker, 'utf8').trim());
      }, 25);
      runner.once('exit', (code) => { clearInterval(tick); clearTimeout(timer); reject(new Error(`runner exited before launch: ${code}`)); });
      runner.once('error', (error) => { clearInterval(tick); clearTimeout(timer); reject(error); });
    });
    assert.doesNotMatch(state.output, /PROFILE=/, 'the profile must be unknown to the teardown: that is the case under test');
    assert.ok(chromePidsFor(profile).length > 0, 'chrome is up while the teardown has no profile to reap');

    await stopRunner(runner, state);

    // SIGTERM, not SIGKILL: only a handled signal runs the runner's own cleanup, so the exit code
    // (128 + 15) is what proves the grace period was given. A SIGKILLed runner reports
    // signalCode 'SIGKILL' and exitCode null, and its detached chrome group is never reaped.
    assert.equal(runner.signalCode, null, 'the runner exited on the signal it was given');
    assert.equal(runner.exitCode, 143, 'so cdp.mjs\'s own exit handler ran the cleanup');
    await assertNoBrowserOrphans(profile);
    assert.equal(existsSync(profile), false, 'and the runner removed the profile it never reported');
    console.log(`PASS  bound trip: no browser processes or profile ${profile}`);
  } finally {
    // Also clears the mess an assertion above just reported (stopRunner is idempotent, so this is
    // the same teardown again — here it is the red-case hygiene, not the thing being asserted).
    await stopRunner(runner, state);
    if (profile) {
      reapTestProfile(profile);
      rmSync(profile, {recursive: true, force: true});
    }
    rmSync(dir, {recursive: true, force: true});
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

test('a launch that fails once is retried, and both attempts are reported (learnings-gib)', {skip: !binary}, async () => {
  // A stub "chromium" that exits immediately: each attempt rejects via the exit guard, so
  // the retry runs without waiting out the endpoint timer, and the surfaced error must name
  // both attempts.
  const dir = mkdtempSync(join(tmpdir(), 'cdp-stub-'));
  const stub = join(dir, 'chromium-stub');
  writeFileSync(stub, '#!/bin/sh\nexit 0\n', {mode: 0o755});
  try {
    await assert.rejects(
      launch({candidates: [stub], endpointTimeoutMs: 1000}),
      (error) => /failed to launch twice: chromium exited early \(0\) \(first attempt: chromium exited early \(0\)\)/.test(error.message),
    );
  } finally {
    rmSync(dir, {recursive: true, force: true});
  }
});

test('a launch whose retry SUCCEEDS recovers: the second attempt endpoint is used (learnings-gib, learnings-25a)', {skip: !binary}, async () => {
  // The recovery branch of the retry: invocation 1 exits 1 (the slow-start failure); invocation
  // 2 prints a devtools endpoint — port 1 refuses connections, so launch() must take wsUrl from
  // attempt 2 and fail LATER, at the websocket step, never with 'failed to launch twice'. The
  // counter file proves attempt 2 actually ran and its endpoint was the one taken.
  const dir = mkdtempSync(join(tmpdir(), 'cdp-stub-'));
  const stub = join(dir, 'chromium-stub');
  const counter = join(dir, 'attempts');
  writeFileSync(stub, '#!/bin/sh\n' +
    `n=$(( \$(cat '${counter}' 2>/dev/null || echo 0) + 1 ))\n` +
    `echo $n > '${counter}'\n` +
    'if [ $n -eq 1 ]; then exit 1; fi\n' +
    "echo 'ws://127.0.0.1:1/devtools/browser/00000000-0000-0000-0000-000000000000'\n" +
    'sleep 300\n', {mode: 0o755});
  try {
    await assert.rejects(
      launch({candidates: [stub], endpointTimeoutMs: 2000}),
      (error) => /devtools websocket failed/.test(error.message) && !/failed to launch twice/.test(error.message),
    );
    assert.equal(readFileSync(counter, 'utf8').trim(), '2', 'the recovered launch must have run exactly two attempts');
  } finally {
    rmSync(dir, {recursive: true, force: true});
  }
});

// learnings-0i3y: the first cut of page.on() fell back to the current session when an event
// carried no id, so browser-level Target.*/Browser.* traffic was delivered to page handlers while
// the comment promised it was not. The rule is a pure function now, and this is its direct test:
// a browser test cannot produce a sessionId-less event (the driver's own send() always attaches
// one, so the browser attributes the reply events to that session too — measured: the browser
// version of this check passed against the buggy fallback).
test('isPageEvent delivers this session only: not browser-level, not another target, not pre-attach', () => {
  const mine = 'SESSION-A';
  assert.equal(isPageEvent({ method: 'Tracing.dataCollected', sessionId: mine }, mine), true, 'the page\'s own event');
  assert.equal(isPageEvent({ method: 'Target.targetCreated' }, mine), false, 'a sessionId-less browser event is not the page\'s');
  assert.equal(isPageEvent({ method: 'Tracing.dataCollected', sessionId: 'SESSION-B' }, mine), false, 'another target\'s event');
  assert.equal(isPageEvent({ method: 'Runtime.consoleAPICalled', sessionId: mine }, null), false, 'nothing before the attach');
  for (const current of [undefined, '']) {
    assert.equal(isPageEvent({ method: 'Runtime.consoleAPICalled', sessionId: mine }, current), false, `no session (${String(current)}) delivers nothing`);
  }
});
