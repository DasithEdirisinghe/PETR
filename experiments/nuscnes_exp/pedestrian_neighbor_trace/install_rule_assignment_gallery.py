"""Install the assignment viewer and its tab without replacing other galleries."""
import argparse
from pathlib import Path


def install(root):
    root = Path(root)
    index = root / 'index.html'
    text = index.read_text()
    target = root / 'rule_assignments'
    target.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name('rule_assignment_gallery.html').read_text()
    if 'id="rule-assignments-tab-button"' not in text:
        button = '<button id="rule-assignments-tab-button" type="button" aria-selected="false">Rule assignments</button>'
        text = text.replace('</nav>', button + '</nav>', 1)
        text = text.replace('<section id="pairs-tab"',
            '<section id="rule-assignments-tab" hidden><iframe id="rule-assignments-frame" class="pair-frame" title="Pedestrian rule assignment audit" data-src="rule_assignments/index.html"></iframe></section>\n<section id="pairs-tab"', 1)
        old = "['detector','pairs','sam','plots','correspondence']"
        if old not in text:
            raise ValueError('Gallery tab controller changed; refusing partial installation')
        text = text.replace(old, "['detector','pairs','sam','plots','correspondence','rule-assignments']", 1)
        text = text.replace("if(tab==='sam')buildSam();",
            "if(tab==='rule-assignments'&&!$('rule-assignments-frame').src)$('rule-assignments-frame').src=$('rule-assignments-frame').dataset.src;if(tab==='sam')buildSam();", 1)
        listener = "$('rule-assignments-tab-button').addEventListener('click',()=>selectTab('rule-assignments'));\nif(location.hash==='#rule-assignments')selectTab('rule-assignments');\n"
        position = text.rfind('</script>')
        if position < 0: raise ValueError('Gallery script not found')
        text = text[:position]+listener+text[position:]
    (target/'index.html').write_text(template)
    index.write_text(text)
    print('Installed:', index, '#rule-assignments')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery-root', default=str(Path(__file__).parent/'outputs/visualizations'))
    install(parser.parse_args().gallery_root)
