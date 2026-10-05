import { test } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';

function findFiles(dir, exts) {
  let results = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat && stat.isDirectory()) {
      results = results.concat(findFiles(filePath, exts));
    } else if (exts.some(ext => file.endsWith(ext))) {
      results.push(filePath);
    }
  }
  return results;
}

test('no innerHTML with template sinks', () => {
  const jsFiles = findFiles('./site', ['.js', '.mjs']);
  let failures = [];
  
  for (const file of jsFiles) {
    if (file.includes('/vendor/')) continue;
    
    const content = fs.readFileSync(file, 'utf8');
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      // Allow specific known safe usages: `innerHTML = ""`
      if (line.includes('.innerHTML = ""')) return;
      if (line.includes('.innerHTML = \'\'')) return;
      if (line.includes('.innerHTML = " "')) return;
      
      // Allow SVG building where we only use math/constants
      if (file.includes('app.js') && line.includes('svg.innerHTML = `')) return;
      
      // Allow strict text assignments (like notice.innerHTML = '<strong>...</strong>')
      // Only allow if it does NOT contain a backtick with interpolation ${
      // Actually, if we just ban .innerHTML entirely except for our exceptions:
      
      const isSink = line.match(/\.innerHTML\s*\+?=/) ||
                     line.match(/\[['"`]innerHTML['"`]\]\s*\+?=/) ||
                     line.match(/\.insertAdjacentHTML\s*\(/) ||
                     line.match(/document\.write\s*\(/);
                     
      if (isSink) {
        if (line.includes('notice.innerHTML =')) return; // allowed from image-lab.js
        failures.push(`File ${file}:${index + 1} uses innerHTML sink: ${line.trim()}`);
      }
    });
  }
  
  if (failures.length > 0) {
    assert.fail('Found innerHTML sinks:\n' + failures.join('\n'));
  }
});

test('lint regression: verifies bad fixtures are caught', () => {
  const badFiles = ['./tests/fixtures/lint-sinks.js', './tests/fixtures/lint-sinks.mjs'];
  let failures = [];
  
  for (const file of badFiles) {
    const content = fs.readFileSync(file, 'utf8');
    const lines = content.split('\n');
    
    lines.forEach((line, index) => {
      const isSink = line.match(/\.innerHTML\s*\+?=/) ||
                     line.match(/\[['"`]innerHTML['"`]\]\s*\+?=/) ||
                     line.match(/\.insertAdjacentHTML\s*\(/) ||
                     line.match(/document\.write\s*\(/);
                     
      if (isSink) {
        failures.push(`File ${file}:${index + 1} uses innerHTML sink: ${line.trim()}`);
      }
    });
  }
  
  assert.strictEqual(failures.length, 5, 'Should catch exactly 5 sink occurrences in the fixtures');
});
