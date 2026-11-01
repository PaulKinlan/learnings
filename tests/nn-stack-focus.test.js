import assert from 'node:assert/strict';
import {launch} from './lib/cdp.mjs';
import {serve} from '../scripts/serve.mjs';

const local = await serve();
const browser = await launch({candidates: ['/usr/bin/chromium']});
const url = local.url + 'neural-networks/';
const selector = '#nn-stack li:has(h3 a)';
const anchor = selector + ' h3 a';
// End states of the 0.15s lift transition on .nn-stack li (chapter.css). A fixed sleep raced
// that transition under load and read the card mid-interpolation (learnings-93m); while a
// transition runs, computed style only ever returns interpolated values, so an exact match on
// the transitioned properties is possible once it has settled — which is what read() waits on.
const lifted = {transform: 'matrix(1, 0, 0, 1, 0, -2)', borderColor: 'rgb(56, 189, 248)',
  boxShadow: 'rgba(0, 0, 0, 0.4) 0px 4px 16px 0px'};
const resting = {transform: 'none', borderColor: 'rgb(100, 116, 139)', boxShadow: 'none'};
const afterClick = process.argv.includes('--before') ? lifted : resting;
async function read(target, label, {focused = false} = {}) {
  // One atomic in-page read: the target match is checked, re-checked one timer tick later
  // (a restarted transition fails the second sample), and the snapshot is captured in the
  // same evaluation. A separate check-then-capture raced hover restarts under load — the
  // predicate saw the settled card and the snapshot then read mid-transition (gate 4b0e5490,
  // learnings-93m). The predicate returns the OBSERVED sample plus a settled flag and selects
  // on `until`, so a timeout message carries the computed styles the card actually showed
  // instead of `last value null` (learnings-ft8).
  return (await browser.waitFor(async (sel, want, focus) => {
    const sample = () => {
      const li = document.querySelector(sel);
      const a = li.querySelector('h3 a');
      const card = getComputedStyle(li);
      const link = getComputedStyle(a);
      return {
        transform: card.transform, borderColor: card.borderColor, boxShadow: card.boxShadow,
        outlineColor: link.outlineColor, outlineStyle: link.outlineStyle,
        focusVisible: a.matches(':focus-visible'), focusWithin: li.matches(':focus-within'),
        hover: li.matches(':hover'), activeElement: document.activeElement === a,
        href: a.href,
      };
    };
    // With `focused`, the read also waits for focus to land on the link: after history
    // navigation the browser restores focus asynchronously, and a style-only settle read
    // the resting card before the restore under load (focusWithin/activeElement false,
    // learnings-a6u4). Polling the focus state with the same bounded waitFor closes that
    // race, the same way the atomic re-check closed the hover race (learnings-93m).
    const settled = (s) => s.transform === want.transform && s.borderColor === want.borderColor &&
      s.boxShadow === want.boxShadow &&
      (!focus || (s.focusWithin && s.activeElement));
    const first = sample();
    if (!settled(first)) return { settled: false, observed: first };
    await new Promise((resolve) => setTimeout(resolve, 60));
    const second = sample();
    return settled(second)
      ? { settled: true, observed: second }
      : { settled: false, observed: second, restarted: true };
  }, { args: [selector, target, focused], timeout: 30000, label, until: (r) => r?.settled === true })).observed;
}
async function away() {
  await browser.send('Input.dispatchMouseEvent', {type: 'mouseMoved', x: 0, y: 0});
}
try {
  await browser.goto(url);
  // 60s bounds with labels on every page wait below: the 15s defaults died under fleet load
  // (observed twice at load average ~10-12, gates 5fc5be1a and a547419d) as an anonymous
  // 'timed out waiting for condition'. waitFor polls, so a healthy box loses nothing
  // (learnings-p564).
  await browser.waitFor(() => Boolean(document.querySelector('#nn-stack li h3 a')),
    {label: 'initial load: nn-stack link to exist', timeout: 60000});
  await away();
  let found = false;
  for (let i = 0; i < 90; i++) {
    await browser.press('Tab');
    if (await browser.evaluate(sel => document.activeElement === document.querySelector(sel), anchor)) {
      found = true;
      break;
    }
  }
  assert.ok(found, 'Tab reaches the in-card anchor');
  await away();
  const keyboard = await read(lifted, 'keyboard focus lift to settle');

  await browser.goto(url);
  await browser.waitFor(() => Boolean(document.querySelector("#nn-stack li h3 a")),
    {label: 'click block: nn-stack link to exist', timeout: 60000});
  await browser.evaluate(() => document.addEventListener('click', event => {
    event.preventDefault();
    event.stopPropagation();
  }, {capture: true}));
  await browser.click(anchor);
  await away();
  const click = await read(afterClick, 'click drop to settle', {focused: true});

  // The back-navigation focus restore comes from the bfcache: Chrome freezes the index
  // page on click and unfreezes it — activeElement included — on back. Under memory
  // pressure the entry can be evicted and the back navigation reloads fresh, where focus
  // is never restored (observed once at load average >4, learnings-a6u4). Detect that via
  // the navigation type of the document we landed on ('back_forward' means a fresh
  // history load; a bfcache restore keeps the original 'navigate' entry) and retry the
  // cycle, so the assertion tests the product rather than the environment.
  let restored = false;
  for (let attempt = 0; attempt < 3 && !restored; attempt++) {
    await browser.goto(url);
    await browser.waitFor(() => Boolean(document.querySelector("#nn-stack li h3 a")),
      {label: 'navigation block: nn-stack link to exist', timeout: 60000});
    const destination = await browser.evaluate(sel => document.querySelector(sel).href, anchor);
    await browser.click(anchor);
    await browser.waitFor(href => location.href === href,
      {args: [destination], label: 'click navigates to the chapter', timeout: 60000});
    const history = await browser.send('Page.getNavigationHistory');
    const previous = history.entries[history.currentIndex - 1];
    assert.ok(previous, 'navigation has a previous page');
    await browser.send('Page.navigateToHistoryEntry', {entryId: previous.id});
    await browser.waitFor(() => Boolean(document.querySelector('#nn-stack li h3 a')),
      {label: 'back navigation: nn-stack link to exist', timeout: 60000});
    restored = await browser.evaluate(() =>
      performance.getEntriesByType('navigation')[0]?.type !== 'back_forward');
  }
  assert.ok(restored, 'back navigation restored the page from the bfcache');
  await away();
  const back = await read(afterClick, 'back-navigation drop to settle', {focused: true});

  await browser.goto(url);
  await browser.waitFor(() => Boolean(document.querySelector("#nn-stack li h3 a")),
    {label: 'hover block: nn-stack link to exist', timeout: 60000});
  await browser.evaluate(sel => document.querySelector(sel).scrollIntoView({block: 'center'}), anchor);
  const position = await browser.evaluate(sel => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return {x: Math.round(b.left + b.width / 2), y: Math.round(b.top + b.height / 2)};
  }, anchor);
  await browser.send('Input.dispatchMouseEvent', {type: 'mouseMoved', ...position});
  const hover = await read(lifted, 'hover lift to settle');
  for (const [name, value] of Object.entries({keyboard, click, back, hover})) console.log(`${name}: ${JSON.stringify(value)}`);
  assert.equal(keyboard.transform, 'matrix(1, 0, 0, 1, 0, -2)');
  assert.equal(keyboard.borderColor, 'rgb(56, 189, 248)');
  assert.equal(keyboard.boxShadow, 'rgba(0, 0, 0, 0.4) 0px 4px 16px 0px');
  assert.equal(keyboard.outlineColor, 'rgb(251, 191, 36)');
  assert.equal(keyboard.outlineStyle, 'solid');
  assert.equal(keyboard.focusVisible, true);
  assert.equal(hover.hover, true);
  assert.equal(hover.transform, keyboard.transform);
  assert.equal(hover.borderColor, keyboard.borderColor);
  assert.equal(hover.boxShadow, keyboard.boxShadow);
  for (const value of [click, back]) {
    assert.equal(value.focusWithin, true, 'clicked link retains focus');
    assert.equal(value.focusVisible, false);
    assert.equal(value.hover, false);
    if (process.argv.includes('--before')) {
      assert.equal(value.transform, keyboard.transform);
    } else {
      assert.equal(value.transform, 'none');
      assert.equal(value.borderColor, 'rgb(100, 116, 139)');
      assert.equal(value.boxShadow, 'none');
    }
  }
} finally {
  await browser.close();
  await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
}
