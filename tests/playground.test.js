import test from 'node:test';
import assert from 'node:assert/strict';

// The playground's Jev engine is a module-private closure (engines.jev.load), so its two
// security boundaries — the 45s AbortSignal and validateAnswers() — are exercised in a real
// browser, the same way image-decision.test.js drives the image lab. Both tests stub the
// network and drive the actual Run button, never the private function directly.

// Select the hosted Jev engine and load it. Loading Jev performs no network request (the fetch
// happens inside decideAll at run time), so the page can be put on Jev before the fetch is stubbed.
async function loadJev(page) {
  await page.evaluate(() => {
    const select = document.querySelector('#engine');
    select.value = 'jev';
    select.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await page.type('#jev-key', 'synthetic-test-not-a-real-key');
  await page.click('#load');
  await page.waitFor(() => document.querySelector('#progress').textContent.includes('Jev ready'), {
    label: 'Jev engine to load',
  });
}

test('playground Jev run cuts off a stalled provider at the 45s boundary and re-enables the button (learnings-uqq)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: ['/usr/bin/chromium'] });
    await page.goto(url + 'decision-models/playground.html');
    await page.waitFor(() => document.querySelector('#engine') !== null);
    await loadJev(page);

    // The run handler's own signal is what is under test, so the interval it asks for is recorded
    // and only the wait is compressed. The provider then accepts the request and never answers, and
    // the abort that ends it is the platform's (the injected signal's reason), not a thrown fixture.
    await page.evaluate(() => {
      window.__timeoutIntervals = [];
      const realTimeout = AbortSignal.timeout.bind(AbortSignal);
      AbortSignal.timeout = (ms) => { window.__timeoutIntervals.push(ms); return realTimeout(50); };
      window.fetch = (resource, init) => new Promise((resolve, reject) => {
        // No signal means no boundary: with the wiring reverted this never settles, which is the hang.
        if (!init.signal) return;
        init.signal.addEventListener('abort', () => reject(init.signal.reason));
      });
    });

    await page.click('.demo .run-demo');
    await page.waitFor(() => window.__timeoutIntervals.length > 0, {
      label: 'the run handler to wire a timeout signal',
    });
    assert.deepEqual(
      await page.evaluate(() => window.__timeoutIntervals),
      [45000],
      "the run must wire the sibling decision lab's 45s boundary",
    );
    await page.waitFor(() => document.querySelector('.demo .results').textContent.includes('Run failed'), {
      label: 'timeout refusal',
    });

    const state = await page.evaluate(() => ({
      results: document.querySelector('.demo .results').textContent,
      answerCount: document.querySelectorAll('.demo .results .answer').length,
      runDisabled: document.querySelector('.demo .run-demo').disabled,
    }));
    assert.equal(state.results, 'Run failed: Request cancelled or timed out. No automatic retry.');
    assert.equal(state.answerCount, 0, 'a timed-out run must not render an answer');
    assert.equal(state.runDisabled, false, 'the timed-out run must terminate and re-enable the run button');
  } finally {
    if (page) await page.close();
    await new Promise((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test('playground Jev run rejects malformed provider answers and still renders valid ones (learnings-wnn)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: ['/usr/bin/chromium'] });
    await page.goto(url + 'decision-models/playground.html');
    await page.waitFor(() => document.querySelector('#engine') !== null);
    await loadJev(page);

    const runState = () => page.evaluate(() => ({
      results: document.querySelector('.demo .results').textContent,
      answerCount: document.querySelectorAll('.demo .results .answer').length,
      runDisabled: document.querySelector('.demo .run-demo').disabled,
    }));

    // A response with no answers object must be refused before render, not throw an uncaught
    // TypeError in the click handler when Object.entries(report.answers) meets undefined.
    await page.evaluate(() => {
      window.fetch = async () => ({ ok: true, status: 200, json: async () => ({ model: 'synthetic-provider' }) });
    });
    await page.click('.demo .run-demo');
    await page.waitFor(() => document.querySelector('.demo .results').textContent.includes('Run failed'), {
      label: 'missing-answers refusal',
    });
    let state = await runState();
    assert.equal(state.results, 'Run failed: Provider did not return an answers object.');
    assert.equal(state.answerCount, 0, 'a response with no answers object must not be rendered');
    assert.equal(state.runDisabled, false, 'the refused run must re-enable the run button');

    // An out-of-set choice is the corrupt render case: before validation, renderAnswer reads
    // probabilities['rm_rf'] (undefined) and prints NaN% as authentic output. It must be refused.
    await page.evaluate(() => {
      window.fetch = async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          model: 'synthetic-provider',
          answers: {
            tool: {
              type: 'choice',
              choice: 'rm_rf',
              confidence: 0.9,
              probabilities: { screenshot_tool: 0.9, read_page: 0.03, history_search: 0.03, download_manager: 0.04 },
            },
          },
        }),
      });
    });
    await page.click('.demo .run-demo');
    await page.waitFor(() => document.querySelector('.demo .results').textContent.includes('Run failed'), {
      label: 'out-of-set-choice refusal',
    });
    state = await runState();
    assert.equal(state.results, 'Run failed: Out-of-set choice: tool.');
    assert.equal(state.answerCount, 0, 'an out-of-set choice must not render a NaN distribution');
    assert.equal(state.runDisabled, false);

    // The gate must not over-block: a contract-shaped response still renders in full.
    await page.evaluate(() => {
      window.fetch = async () => ({
        ok: true,
        status: 200,
        json: async () => ({
          model: 'jev-1.13.0',
          answers: {
            tool: {
              type: 'choice',
              choice: 'screenshot_tool',
              confidence: 0.9,
              probabilities: { screenshot_tool: 0.9, read_page: 0.03, history_search: 0.03, download_manager: 0.04 },
            },
          },
          usage: { input_tokens: 42 },
        }),
      });
    });
    await page.click('.demo .run-demo');
    await page.waitFor(() => document.querySelector('.demo .results .answer') !== null, {
      label: 'valid answer render',
    });
    state = await runState();
    assert.equal(state.answerCount, 1, 'a valid response renders exactly one answer');
    assert.match(state.results, /choice: screenshot_tool/);
    assert.match(state.results, /42 input tokens/);
    assert.doesNotMatch(state.results, /Run failed/);
  } finally {
    if (page) await page.close();
    await new Promise((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});
