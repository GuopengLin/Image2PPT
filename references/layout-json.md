# Layout JSON Reference

Use layout JSON when rebuilding slide images as editable PPTX files with `scripts/deck/build_pptx_from_layout.py`.

## Top Level

```json
{
  "slide_size": { "width_in": 13.333333, "height_in": 7.5 },
  "source_width": 1182,
  "source_height": 665,
  "background": "#FFFFFF",
  "slides": [
    {
      "background": "#FFFFFF",
      "elements": []
    }
  ]
}
```

Coordinates are in source-image pixels. The script scales them to the PowerPoint slide size.

For a one-slide layout, `elements` may be placed at the top level instead of under `slides`.

## Elements

### Text

```json
{
  "type": "text",
  "name": "slide-title",
  "text": "目  录",
  "box": [532, 138, 122, 58],
  "font": "KaiTi",
  "size": 31,
  "bold": true,
  "color": "#481258",
  "align": "center",
  "valign": "middle",
  "line_spacing": 1.05
}
```

Use editable text for all normal slide copy. Preserve line breaks from the reference image when they are part of layout.

### Image

```json
{
  "type": "image",
  "name": "icon-document",
  "path": "assets/page_002/18_icon_document.png",
  "box": [112, 332, 51, 59]
}
```

Use images for extracted logos, icons, background ornaments, line art, photos, and complex decorative marks.

### Shape

```json
{
  "type": "shape",
  "name": "catalog-card-01",
  "shape": "rounded_rect",
  "box": [64, 225, 146, 306],
  "fill": "transparent",
  "line": "#BCB9BE",
  "line_width": 1,
  "radius": 0.06
}
```

Supported shapes include `rect`, `rounded_rect`, `oval`, `diamond`, `triangle`, and `trapezoid`.

### Line

```json
{
  "type": "line",
  "name": "divider",
  "points": [861, 329, 1104, 329],
  "line": "#D6D6D6",
  "line_width": 0.7,
  "dash": "dash"
}
```

Use native lines for separators, rules, and diagram connectors when they are simple. Use extracted images for complex gradient/ornamental lines.

`points` accepts two encodings:

- Flat `[x1, y1, x2, y2]` — one straight segment (a PPT straight connector).
- Nested `[[x, y], [x, y], ...]` with 2+ vertices — an open polyline (e.g. an elbow connector), emitted as a freeform shape with no fill.

Optional fields:

- `dash`: `"dash"`/`"dashed"` or `"dot"`/`"dotted"`.
- `arrow_start` / `arrow_end`: `true` adds a triangle arrowhead at that end (`points` order defines which end is which).
- `box`: `[x, y, w, h]` bbox of the points — include it on auto-extracted connectors so containment-based z-ordering can place the line above its background/card.

The auto pipeline emits these for inventory `connector` components whose stroke geometry fits a straight or elbow model confidently (see `scripts/page/layout/connector_geometry.py`); anything else keeps the transparent-PNG fallback.

## Ordering

Elements render in list order. Put background bands and large decorative images first, then frames/shapes, then icons, then text.

## Batch Layouts

For many pages, create one `page_###.layout.json` per source image. Combine them with:

```bash
python scripts/deck/combine_layouts.py --layouts layouts --out combined.layout.json
python scripts/deck/build_pptx_from_layout.py --layout combined.layout.json --assets-root . --out output/deck.pptx
```
