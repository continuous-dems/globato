"""Offline HTML comparison of existing review PNGs; never part of the hook."""

from html import escape
import json
from pathlib import Path


def publish(run, stages, complete=False, details=""):
    available = []
    for stage in stages:
        if stage.get("image"):
            available.append(stage)
        if stage.get("overlay"):
            available.append(
                dict(
                    label=stage["label"] + " — removed / restored sites",
                    status=stage["status"],
                    image=stage["overlay"],
                )
            )
    options = "".join(
        f'<option value="{i}">{escape(s["label"])}</option>'
        for i, s in enumerate(available)
    )
    rows = []
    for stage in stages:
        links = []
        for field, label in (
            ("image", "Color hillshade"),
            ("overlay", "Rejected and restored locations"),
        ):
            if stage.get(field):
                links.append(
                    f'<a href="{escape(stage[field], quote=True)}">{label}</a>'
                )
        for label, path in stage.get("downloads", []):
            links.append(f'<a href="{escape(path, quote=True)}">{escape(label)}</a>')
        rows.append(
            f"<tr><td>{escape(stage['label'])}</td><td>{escape(stage['status'])}</td><td>{' · '.join(links) or 'Pending'}</td></tr>"
        )
    payload = json.dumps(available).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
REFRESH<title>Stack QC review</title>
<style>
body{font:16px system-ui,sans-serif;color:#192d3a;background:#f5f7f8;margin:0 auto;padding:24px;max-width:1600px}
h1{margin:0 0 8px}p{line-height:1.5;max-width:1100px}a{color:#075da0}
.status{padding:12px;background:#e2eef3;border-radius:6px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:24px 0}
figure{margin:0;background:white;padding:12px;border:1px solid #ccd8df;border-radius:6px}select{width:100%;font:inherit;padding:8px;margin-bottom:8px}img{width:100%;height:auto;display:block}figcaption{padding-top:8px;line-height:1.4}
table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #d9e0e5}th{background:#e9eff2}
@media(max-width:800px){.compare{grid-template-columns:1fr}body{padding:12px}}
</style>
<h1>Review</h1><p class="status">STATUS</p><p>DETAILS</p>
<p>Select two stages to compare. All hillshades use the original DEM’s color limits,
315° illumination and the same map extent. Click either image to open it at full size.
Red marks rejected source-cell locations. Black marks sites finally restored; the black
symbols are deliberately larger than the cells so they remain visible over red.
The enlarged symbols do not mean a larger area was restored. On earlier pass maps,
black shows which rejected sites were eventually restored, not when restoration
occurred. The restored-only image colors those sites by the final measured elevation.
The all-measured image colors every cell with a final measurement from this module.
Gray areas without colored cells have no measurement in that cleaned module stack,
even if the combined Globato DEM later fills them.</p>
<div class="compare"><figure><select id="left" aria-label="Left comparison stage">OPTIONS</select><a id="leftLink"><img id="leftImage" alt="Left selected stage"></a><figcaption id="leftCaption"></figcaption></figure>
<figure><select id="right" aria-label="Right comparison stage">OPTIONS</select><a id="rightLink"><img id="rightImage" alt="Right selected stage"></a><figcaption id="rightCaption"></figcaption></figure></div>
<table><thead><tr><th>Stage</th><th>Surface / status</th><th>Open maps and GeoTIFFs</th></tr></thead><tbody>ROWS</tbody></table>
<p>The last removal pass goes to the final cleaned measured stack for this module.
When needed, the review also builds a cheap module-only interpolated preview from
that cleaned stack. This preview is only for before/after interpretation. The final
combined Globato DEM remains the production product and may include other modules.
Where measurements overlap, a restored site is shown at its final combined measured
elevation within this module stack, which can also include other accepted source
contributions from the same module. Source-cell counts in the log are not the same
as distinct grid-cell locations on these maps.</p>
<script>
const stages=PAYLOAD, saved=new URLSearchParams(location.hash.slice(1));
function update(side){const i=Number(document.getElementById(side).value),s=stages[i];if(!s)return;
document.getElementById(side+'Image').src=s.image;document.getElementById(side+'Link').href=s.image;
document.getElementById(side+'Caption').textContent=s.label+' — '+s.status;}
for(const side of ['left','right']){const select=document.getElementById(side);let i=Number(saved.get(side)??(side==='left'?0:Math.max(0,stages.length-1)));
select.value=String(Math.max(0,Math.min(i,stages.length-1)));select.addEventListener('change',()=>{update(side);location.hash=new URLSearchParams({left:document.getElementById('left').value,right:document.getElementById('right').value});});update(side);}
</script></html>"""
    values = dict(
        REFRESH="" if complete else '<meta http-equiv="refresh" content="15">',
        STATUS="Complete — all available stages linked below."
        if complete
        else "Running — this page refreshes every 15 seconds as stages complete.",
        DETAILS=escape(details),
        OPTIONS=options,
        ROWS="".join(rows),
        PAYLOAD=payload,
    )
    import re

    page = re.sub(
        r"\b(?:REFRESH|STATUS|DETAILS|OPTIONS|ROWS|PAYLOAD)\b",
        lambda match: values[match.group()],
        page,
    )
    run = Path(run)
    temporary = run / "review.html.partial"
    temporary.write_text(page)
    temporary.replace(run / "review.html")
