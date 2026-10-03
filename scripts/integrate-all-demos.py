import re
from pathlib import Path

BASE_DIR = Path("/home/paulkinlan/learnings/site/neural-networks")
LAB_PATH = BASE_DIR / "lab.html"

with open(LAB_PATH, "r", encoding="utf-8") as f:
    lab_html = f.read()

def extract_section(section_id):
    pattern = re.compile(
        rf'(<section id="{section_id}" class="nn-section">.*?</section>\s*(?:<div id="[^"]*stepper"></div>\s*)?(?:<div class="section-code-widget"[^>]*></div>\s*)?)(?=\s*<!-- =|\s*<section id=|\s*</main>)',
        re.DOTALL
    )
    m = pattern.search(lab_html)
    if m:
        return m.group(1).strip()
    return None

SECTIONS = {
    'mlp-backprop': extract_section('mlp-backprop'),
    'cnn-lab': extract_section('cnn-lab'),
    'rnn-lab': extract_section('rnn-lab'),
    'deep-resnet-lab': extract_section('deep-resnet-lab'),
    'transformer-lab': extract_section('transformer-lab'),
    'diffusion-lab': extract_section('diffusion-lab'),
    'decision-lab': extract_section('decision-lab'),
    'block-builder': extract_section('block-builder'),
}

for k, v in SECTIONS.items():
    if not v:
        raise RuntimeError(f"Failed to extract section {k}")
    print(f"Extracted {k}: {len(v)} bytes")

def integrate_into_page(page_rel_path, section_keys, is_nested=False):
    page_path = BASE_DIR / page_rel_path
    with open(page_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Ensure CSP allows wasm-unsafe-eval for the kernels
    html = html.replace("script-src 'self';", "script-src 'self' 'wasm-unsafe-eval';")

    # Add nn.css link if not present
    nn_css_rel = "../nn.css" if is_nested else "./nn.css"
    if nn_css_rel not in html:
        css_tag = f'  <link rel="stylesheet" href="{nn_css_rel}">\n'
        html = html.replace('  <link rel="stylesheet" href="./chapter.css">\n', f'  <link rel="stylesheet" href="./chapter.css">\n{css_tag}')
        if is_nested:
            html = html.replace('  <link rel="stylesheet" href="../chapter.css">\n', f'  <link rel="stylesheet" href="../chapter.css">\n{css_tag}')

    # Combine extracted sections
    injected_sections = "\n\n    <!-- =====================================================================\n"
    injected_sections += "         INTEGRATED FULL TRAINERS & LIVE BENCHMARK DEMOS\n"
    injected_sections += "         ===================================================================== -->\n\n"
    for sk in section_keys:
        injected_sections += SECTIONS[sk] + "\n\n"

    # Insert before <section class="prose"><h2>Sources
    sources_marker = '<section class="prose"><h2>Sources'
    if sources_marker in html:
        html = html.replace(sources_marker, f"{injected_sections}    {sources_marker}")
    else:
        # insert before </article>
        article_end = '</article>'
        if article_end in html:
            html = html.replace(article_end, f"{injected_sections}    {article_end}")

    # Add app.js script if not present
    app_js_rel = "../app.js" if is_nested else "./app.js"
    if app_js_rel not in html:
        script_tag = f'  <script type="module" src="{app_js_rel}"></script>\n'
        html = html.replace('</body>', f'{script_tag}</body>')

    with open(page_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Successfully integrated {section_keys} into {page_rel_path}")

# 1. architectures.html gets MLP, RNN, Deep ResNet, and Diffusion
integrate_into_page("architectures.html", ["mlp-backprop", "rnn-lab", "deep-resnet-lab", "diffusion-lab"])

# 2. cnn.html gets cnn-lab
integrate_into_page("cnn.html", ["cnn-lab"])

# 3. transformer.html gets transformer-lab
integrate_into_page("transformer.html", ["transformer-lab"])

# 4. loss-and-calibration.html gets decision-lab
integrate_into_page("loss-and-calibration.html", ["decision-lab"])

# 5. modern-advancements.html gets block-builder
integrate_into_page("modern-advancements.html", ["block-builder"])

print("All chapter subpages deeply integrated with full live trainers!")
