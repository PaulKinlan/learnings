import assert from 'node:assert/strict';
import {launch} from './lib/cdp.mjs';
import {serve} from '../scripts/serve.mjs';

const local = await serve();
const browser = await launch({candidates: ['/usr/bin/chromium']});
const url = local.url + 'neural-networks/';
const selector = '#nn-stack li:has(h3 a)';
const anchor = selector + ' h3 a';
const settle = () => new Promise(resolve => setTimeout(resolve, 250));
async function read() {
  await settle();
  return browser.evaluate(sel => {
    const li = document.querySelector(sel);
    const a = li.querySelector('h3 a');
    const card = getComputedStyle(li);
    const link = getComputedStyle(a);
    return {transform: card.transform, borderColor: card.borderColor, boxShadow: card.boxShadow,
      outlineColor: link.outlineColor, outlineStyle: link.outlineStyle,
      focusVisible: a.matches(':focus-visible'), focusWithin: li.matches(':focus-within'),
      hover: li.matches(':hover'), activeElement: document.activeElement === a,
      href: a.href};
  }, selector);
}
async function away() {
  await browser.send('Input.dispatchMouseEvent', {type: 'mouseMoved', x: 0, y: 0});
}
try {
  await browser.goto(url);
  await browser.waitFor(() => Boolean(document.querySelector('#nn-stack li h3 a')));
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
  const keyboard = await read();

  await browser.goto(url);
  await browser.waitFor(() => Boolean(document.querySelector("#nn-stack li h3 a")));
  await browser.evaluate(() => document.addEventListener('click', event => {
    event.preventDefault();
    event.stopPropagation();
  }, {capture: true}));
  await browser.click(anchor);
  await away();
  const click = await read();

  await browser.goto(url);
  await browser.waitFor(() => Boolean(document.querySelector("#nn-stack li h3 a")));
  const destination = await browser.evaluate(sel => document.querySelector(sel).href, anchor);
  await browser.click(anchor);
  await browser.waitFor(href => location.href === href, {args: [destination]});
  const history = await browser.send('Page.getNavigationHistory');
  const previous = history.entries[history.currentIndex - 1];
  assert.ok(previous, 'navigation has a previous page');
  await browser.send('Page.navigateToHistoryEntry', {entryId: previous.id});
  await browser.waitFor(() => Boolean(document.querySelector('#nn-stack li h3 a')));
  await away();
  const back = await read();

  await browser.goto(url);
  await browser.waitFor(() => Boolean(document.querySelector("#nn-stack li h3 a")));
  await browser.evaluate(sel => document.querySelector(sel).scrollIntoView({block: 'center'}), anchor);
  const position = await browser.evaluate(sel => {
    const b = document.querySelector(sel).getBoundingClientRect();
    return {x: Math.round(b.left + b.width / 2), y: Math.round(b.top + b.height / 2)};
  }, anchor);
  await browser.send('Input.dispatchMouseEvent', {type: 'mouseMoved', ...position});
  const hover = await read();
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
