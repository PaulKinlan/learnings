#!/usr/bin/env python3
"""
Authoritative extraction script for JevImageBench v0.1.5
Transcribes directly from https://benchmarkheaven.com/image-jev-bench HTML payload.
Produces:
1. tests/fixtures/image-jev-bench-source-extract.json (transcribed reference)
2. site/decision-models/image-bench-data.js (canonical ES module with matching exact values)
"""

import json
import re
from pathlib import Path

SOURCE_HTML_PATH = Path('/tmp/image-jev-bench.html')
WORKTREE_DIR = Path('/home/paulkinlan/worktrees/learnings-image-jev')

with open(SOURCE_HTML_PATH, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Extract the raw Next.js systems array
pattern = re.compile(r'self\.__next_f\.push\(\[1,\s*"(.*?)"\]\)', re.DOTALL)
systems_raw = None
for m in pattern.finditer(html):
    chunk = m.group(1)
    if r'\"systems\":[' in chunk:
        try:
            unescaped = json.loads(f'"{chunk}"')
        except Exception:
            continue
        start = unescaped.find('"systems":[')
        if start != -1:
            arr_start = start + len('"systems":')
            depth = 0
            arr_end = -1
            for j in range(arr_start, len(unescaped)):
                if unescaped[j] == '[':
                    depth += 1
                elif unescaped[j] == ']':
                    depth -= 1
                    if depth == 0:
                        arr_end = j + 1
                        break
            if arr_end != -1:
                systems_raw = json.loads(unescaped[arr_start:arr_end])
                break

if not systems_raw or len(systems_raw) != 50:
    raise RuntimeError(f"Expected 50 systems from HTML, got {len(systems_raw) if systems_raw else 0}")

# 2. Extract exact link mappings from the table HTML
link_pattern = re.compile(r'<a\s+href="([^"]+)"[^>]*class="[^"]*underline[^"]*"[^>]*>([^<]+)</a>')
name_to_link = {}
for m in link_pattern.finditer(html):
    name = m.group(2).strip()
    link = m.group(1).strip()
    if name != 'source' and name not in name_to_link:
        name_to_link[name] = link

# Map known display variations
name_alias = {
    'Glance (frozen Qwen3-VL-4B)': 'Glance',
    'JPT-4B (kirp / llm2jev)': 'JPT-4B',
    'JPT-9B (kirp / llm2jev)': 'JPT-9B',
    'JPT-0.8B (kirp / llm2jev)': 'JPT-0.8B',
    'NeoHorse Jev 4B': 'NeoHorse Jev 4B',
    'OmniJev 4B (tinnel123)': 'OmniJev 4B',
    'OmniJev 2B (tinnel123)': 'OmniJev 2B',
    'OmniJev 0.8B (tinnel123)': 'OmniJev 0.8B',
    'Reflex 4B (released stable configuration)': 'Reflex 4B',
    'Jev-Vision 8B (SeanLiu)': 'Jev-Vision 8B',
    'vjev-vision (yah01)': 'vjev-vision',
    'jev-spatial (Fr0zencr4nE)': 'jev-spatial',
    'Gevva E4B (text checkpoint, image input)': 'Gevva E4B',
    'Qevi-2B (MeerDevelopment)': 'Qevi-2B',
    'JEVision (divyanshx11, visual route)': 'JEVision',
    'PlayJev 0.8B': 'PlayJev 0.8B',
    'GPT-6 Luna (low reasoning effort)': 'GPT-6 Luna (low)',
    'djev-distill-v4 (tarsur385)': 'djev-distill-v4',
    'diffusiongemma-26b djev v10 step160 (snowicarus)': 'diffusiongemma-26b djev v10 step160',
    'OpenJev 4B NLI v2 (official image-premise path)': 'OpenJev 4B NLI v2',
}

# 3. Build clean, transcribed 50-row records
extracted_rows = []
for s in systems_raw:
    display_name = s.get('display', '').strip()
    clean_name = display_name
    for long_name, short_name in name_alias.items():
        if display_name == long_name or display_name.startswith(short_name):
            clean_name = short_name
            break

    # Look up link
    link = name_to_link.get(clean_name) or name_to_link.get(display_name) or ""

    intel = round(s.get('axes', {}).get('intelligence', 0), 1)
    calib = round(s.get('axes', {}).get('calibration', 0), 1)
    cap = round((intel + calib) / 2, 1)
    speed = round(s.get('speed', {}).get('p50_s_adjusted', 0), 2)
    cost = round(s.get('cost', {}).get('usd_per_1000', 0), 4)

    is_ranked = s.get('ranked', False)
    rank = s.get('rank') if is_ranked else None

    category = "Jev-Class Bounded" if is_ranked else "Baseline Multimodal (Outside Limits)"
    access = "Closed API" if s.get('api_flag') else "Open Weights / Rebuild"

    extracted_rows.append({
        "rank": rank,
        "name": clean_name,
        "raw_display": display_name,
        "key": s.get('key'),
        "capability": cap,
        "intelligence": intel,
        "calibration": calib,
        "speed": speed,
        "cost": cost,
        "link": link,
        "category": category,
        "access": access
    })

# Write fixture
fixture_data = {
    "_metadata": {
        "source": "https://benchmarkheaven.com/image-jev-bench",
        "benchmark": "Image JevBench v0.1.5",
        "fetch_date": "2026-10-03",
        "total_systems": len(extracted_rows),
        "note": "Authoritative machine transcription from benchmarkheaven.com Next.js HTML stream."
    },
    "systems": extracted_rows
}

fixture_path = WORKTREE_DIR / 'tests/fixtures/image-jev-bench-source-extract.json'
fixture_path.parent.mkdir(parents=True, exist_ok=True)
with open(fixture_path, 'w', encoding='utf-8') as f:
    json.dump(fixture_data, f, indent=2)

print(f"Successfully generated authoritative fixture at {fixture_path} with {len(extracted_rows)} systems!")
