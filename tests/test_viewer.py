import re

from geoai04.config import load_config
from geoai04.pipeline import run_pipeline
from geoai04.viewer import build_split_view_html, render_pair_previews

import rasterio


def test_viewer_is_fully_offline(synthetic_pair, tmp_path):
    res = run_pipeline(synthetic_pair["before"], synthetic_pair["after"], load_config(), out_root=tmp_path, run_name="v")
    b, a, psize, fsize = render_pair_previews(synthetic_pair["before"], synthetic_pair["after"], max_dim=400)
    with rasterio.open(synthetic_pair["before"]) as s:
        tr = s.transform
    html = build_split_view_html(b, a, res.polygons, tr, psize, fsize)
    assert html.count('class="poly"') == 3
    urls = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert urls == []
    text = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html).lower()  # ignore image payloads
    assert "cdn" not in text and "unpkg" not in text and "<link" not in text
