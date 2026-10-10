import test from 'node:test';
import assert from 'node:assert/strict';

test('attention playground roving tabindex, arrow navigation, and hover status stability (learnings-5us)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  let page;
  try {
    page = await launch({ candidates: ['/usr/bin/chromium'] });
    await page.goto(url + 'neural-networks/transformer.html');
    await page.waitFor(() => document.querySelector('[data-widget="attention"] table[data-heatmap]') !== null, {
      label: 'attention playground heatmap table to mount',
    });

    const getGridState = () => page.evaluate(() => {
      const table = document.querySelector('[data-widget="attention"] table[data-heatmap]');
      const buttons = Array.from(table.querySelectorAll('tbody button'));
      const status = document.querySelector('[data-widget="attention"] [data-status]')?.textContent;
      return {
        role: table.getAttribute('role'),
        rowCount: table.getAttribute('aria-rowcount'),
        colCount: table.getAttribute('aria-colcount'),
        count: buttons.length,
        tabIndexZeros: buttons.filter((b) => b.tabIndex === 0).length,
        tabIndexMinusOnes: buttons.filter((b) => b.tabIndex === -1).length,
        zeroIndex: buttons.findIndex((b) => b.tabIndex === 0),
        activeIndex: buttons.findIndex((b) => b === document.activeElement),
        activeAria: document.activeElement?.getAttribute('aria-label') ?? null,
        status,
      };
    });

    // 1. Invariant at init: role=grid semantics, exactly 1 tabIndex=0 button out of 36
    const init = await getGridState();
    assert.equal(init.count, 36, 'grid must contain exactly 36 cell buttons (6x6)');
    assert.equal(init.role, 'grid', 'table must have role="grid"');
    assert.equal(init.rowCount, '7', 'grid must declare aria-rowcount 7 (1 header + 6 data rows)');
    assert.equal(init.colCount, '7', 'grid must declare aria-colcount 7 (1 header + 6 data cols)');
    assert.equal(init.tabIndexZeros, 1, 'exactly one of the 36 cell buttons must have tabindex=0 at init');
    assert.equal(init.tabIndexMinusOnes, 35, 'the remaining 35 cell buttons must have tabindex=-1 at init');
    assert.equal(init.zeroIndex, 0, 'initial roving tabindex must be on cell (0, 0)');
    assert.match(init.status, /^The \(1\) → The \(1\):/, 'initial status readout reflects selected query');

    // 2. Tab navigation: Tab enters the grid once, next Tab leaves it (not 36 stops)
    await page.evaluate(() => {
      document.querySelector('[data-widget="attention"] [data-control="mask"]').focus();
    });
    await page.press('Tab');
    const entered = await getGridState();
    assert.equal(entered.activeIndex, 0, 'Tab from preceding control must focus the single roving tab stop');
    assert.equal(entered.tabIndexZeros, 1, 'exactly one button retains tabindex=0 upon entry');

    await page.press('Tab');
    const leftGrid = await page.evaluate(() => {
      const active = document.activeElement;
      return {
        inHeatmap: Boolean(active?.closest('[data-heatmap]')),
        tagName: active?.tagName,
      };
    });
    assert.equal(leftGrid.inHeatmap, false, 'subsequent Tab must exit the grid rather than stopping at other cells');

    // 3. Keyboard navigation: ArrowRight / ArrowDown move focus and update role=status readout
    await page.evaluate(() => {
      document.querySelector('[data-widget="attention"] [data-heatmap] tbody button').focus();
    });
    const beforeRight = await getGridState();
    assert.equal(beforeRight.activeIndex, 0);

    // ArrowRight moves focus to the next column and updates role=status
    await page.press('ArrowRight');
    const afterRight = await getGridState();
    assert.equal(afterRight.activeIndex, 1, 'ArrowRight must move focus to next column (0, 1)');
    assert.equal(afterRight.zeroIndex, 1, 'roving tabindex must follow focus to index 1');
    assert.equal(afterRight.tabIndexZeros, 1, 'exactly one cell button must have tabindex=0 after ArrowRight');
    assert.notEqual(afterRight.status, beforeRight.status, 'ArrowRight must update role=status readout');
    assert.match(afterRight.status, /^The \(1\) → animal \(2\):/, 'status readout reflects cell (0, 1)');

    // ArrowDown moves focus down one row and updates role=status
    await page.press('ArrowDown');
    const afterDown = await getGridState();
    assert.equal(afterDown.activeIndex, 7, 'ArrowDown must move focus down one row (1, 1)');
    assert.equal(afterDown.zeroIndex, 7, 'roving tabindex must follow focus to index 7');
    assert.equal(afterDown.tabIndexZeros, 1, 'exactly one cell button must have tabindex=0 after ArrowDown');
    assert.notEqual(afterDown.status, afterRight.status, 'ArrowDown must update role=status readout');
    assert.match(afterDown.status, /^animal \(2\) → animal \(2\):/, 'status readout reflects cell (1, 1)');

    // 4. Clamping at edges: ArrowRight at the last column stays put
    await page.press('End');
    const atRowEnd = await getGridState();
    assert.equal(atRowEnd.activeIndex, 11, 'End key must jump to last column in row (index 11)');
    assert.match(atRowEnd.status, /^animal \(2\) → street \(6\):/);

    await page.press('ArrowRight');
    const clampedCol = await getGridState();
    assert.equal(clampedCol.activeIndex, 11, 'ArrowRight at last column must clamp and stay put');
    assert.equal(clampedCol.zeroIndex, 11, 'roving tabindex must remain at last column');
    assert.equal(clampedCol.tabIndexZeros, 1, 'still exactly one tabindex=0 button when clamped');
    assert.equal(clampedCol.status, atRowEnd.status, 'readout must remain unchanged when clamped at edge');

    // 5. Mouse move over a cell does NOT change the status readout text
    const targetCell = await page.evaluate(() => {
      const rows = document.querySelectorAll('[data-widget="attention"] [data-heatmap] tbody tr');
      // Cell (3, 2): row 3 ("didn't"), col 2 ("animal") -> cell button index 20
      const cell = rows[3].querySelectorAll('button')[2];
      cell.scrollIntoView({ block: 'center', inline: 'center' });
      const box = cell.getBoundingClientRect();
      return {
        x: Math.round(box.x + box.width / 2),
        y: Math.round(box.y + box.height / 2),
      };
    });
    const statusBeforeHover = clampedCol.status;
    await page.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: targetCell.x, y: targetCell.y });
    await new Promise((resolve) => setTimeout(resolve, 100));
    const afterHover = await getGridState();
    assert.equal(afterHover.status, statusBeforeHover, 'mouse move over a cell must not change status readout text');

    // 6. Click on a cell moves focus and preserves exactly one tabindex=0 button
    const cellSelector = '[data-widget="attention"] [data-heatmap] tbody tr:nth-child(4) td:nth-child(4) button';
    await page.click(cellSelector);
    const afterClick = await getGridState();
    assert.equal(afterClick.activeIndex, 20, 'clicked cell must receive focus');
    assert.equal(afterClick.zeroIndex, 20, 'clicked cell must become the roving tabindex=0 stop');
    assert.equal(afterClick.tabIndexZeros, 1, 'still exactly one cell button with tabindex=0 after click');
    assert.equal(afterClick.tabIndexMinusOnes, 35, 'remaining 35 cell buttons have tabindex=-1 after click');
    assert.match(afterClick.status, /^cross \(4\) → didn't \(3\):/, 'click must update status readout for target cell');
  } finally {
    if (page) await page.close();
    if (server) await new Promise((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});
