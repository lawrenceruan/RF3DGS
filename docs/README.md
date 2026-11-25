# RF3DGS academic project homepage

Static project page with the manuscript, original result figures, interactive comparisons, method overview, benchmark tables, installation and usage instructions, and project updates.

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000.

## GitHub Pages

In repository **Settings → Pages**, choose **Deploy from a branch**, branch **main**, folder **/docs**, and save. The expected project URL is https://lawrenceruan.github.io/RF3DGS/ once Pages has deployed successfully.

No build step or third-party web dependencies are required. All asset links are relative, so the page works under the repository URL prefix.

## Editing

- `index.html`: paper content, updates, tables, and usage guide.
- `style.css`: responsive academic layout.
- `app.js`: comparison sliders and copy buttons.
- `assets/run_scene.sh`: example workflow; run from the RF3DGS root in a configured GPU environment.

The example commands were checked against the public import `00e7b56`; they have not been validated with a new end-to-end training run. Internal Superpod improvements are explicitly distinguished from the public snapshot.

## Figure provenance

Figures are extracted from the supplied RF3DGS manuscript: teaser (page 2), method (page 4), real-world comparisons (page 10), and synthetic comparisons/ablation (page 12). Comparison-slider images are corresponding input and transmission crops from the real-world figure. The quantitative tables reproduce page 11. The PDF is included unchanged. Author names are omitted pending the final author list.
