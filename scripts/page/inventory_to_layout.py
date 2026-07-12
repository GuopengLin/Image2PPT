#!/usr/bin/env python
"""Convert an inventory JSON into asset manifest + layout JSON.

For each inventory entry:
  - "text":  → text element in layout. Auto-detect colour (sampled from
    the source image), bold (stroke-density ratio), and font size
    (bbox height × type-aware multiplier).
  - "image": → asset manifest crop spec + image element in layout.

Z-order: image elements come FIRST in layout (rendered behind), text
elements come LAST (rendered on top, never covered by images). All
text shapes get transparent fill via build_pptx_from_layout.

This file is now a thin facade. The implementation lives in the
`layout/` sub-package, one module per concern. External code keeps
importing names from `inventory_to_layout` — they are re-exported here.

Usage:
    python inventory_to_layout.py --inventory inv.json --source slide.png \\
        --asset-prefix assets/page_001 \\
        --cleaned clean.png \\
        --out-manifest m.json --out-layout l.json --out-assets-dir assets/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parent           # scripts/page
SCRIPTS_ROOT = PAGE_DIR.parent                        # scripts
for _p in (PAGE_DIR, SCRIPTS_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---- Public API re-exports ---------------------------------------------
# Names other modules / tests import as `from inventory_to_layout import X`.
# Keep this block stable so the facade contract does not change as the
# `layout/` sub-package evolves.
from fontconfig_helper import fontconfig_font_path as _fontconfig_font_path  # noqa: E402, F401
from shared.bg_sample import (  # noqa: E402, F401
    estimate_canvas_hex as estimate_slide_background_hex,
    sample_background_for_bbox as _sample_background_for_bbox,
)
from shared.geometry import (  # noqa: E402, F401
    bgr_to_hex as _bgr_to_hex,
    connector_on_container_border as _connector_on_container_border,
    intersection_area as _intersection_area,
    overlaps_text as _overlaps_text,
)

from layout.bbox_shape import (  # noqa: E402, F401
    _pad_icon_bbox,
    _trim_short_stat_text_bbox,
)
from layout.bullet_images import (  # noqa: E402, F401
    _detect_leading_dot_marker,
    _marker_has_neighbour,
    _marker_rgba_from_source,
    restore_ignored_bullet_marker_images,
)
from layout.color import (  # noqa: E402, F401
    _classify_text_color,
    _color_close,
    _hex_to_rgb,
    _normalize_run_colors,
    _sampled_color_hex,
)
from layout.grouping import unify_group_sizes  # noqa: E402, F401
from layout.icon_alpha import (  # noqa: E402, F401
    _ICON_PAD_ROLES,
    _SPARSE_VISUAL_ROLES,
    _line_art_alpha,
    _scrub_text_boxes_from_icon_crop,
    _simple_background_strip,
)
from layout.lists import (  # noqa: E402, F401
    _BULLET_PREFIX_CHARS,
    _ORDERED_PREFIX_RE,
    _list_marker_geometry,
    _list_prefix,
    strip_leading_list_markers,
)
from layout.outline import (  # noqa: E402, F401
    _outline_should_be_native_shape,
    _outline_should_keep_full_crop,
    _sample_card_fill_color,
    _sample_outline_color,
    _split_filled_outline_rows,
)
from layout.render_fit import (  # noqa: E402, F401
    _binary_size_candidates,
    fit_text_render,
)
from layout.render_font import (  # noqa: E402, F401
    _FONT_CACHE,
    _FONT_CANDIDATES_CACHE,
    _TEXT_RENDER_CACHE,
    _font_candidates,
    _load_font,
    _render_text_metrics,
    default_ppt_font,
)
from layout.render_ink import (  # noqa: E402, F401
    _mask_shape_error,
    _target_text_metrics,
)
from layout.text_emit import _emit_text_element_record  # noqa: E402, F401
from layout.text_mixed import (  # noqa: E402, F401
    _build_runs_from_char_sizes,
    _char_colors_from_runs,
    _glyph_height_in_char_box,
    _label_prefix_split_index,
    mixed_size_runs_from_char_boxes,
)
from layout.text_runs import _per_char_runs  # noqa: E402, F401
from layout.text_sizing import (  # noqa: E402, F401
    CAPTURE_FACTOR_LONG,
    CAPTURE_FACTOR_MEDIUM,
    CAPTURE_FACTOR_SHORT,
    _FONT_LADDER,
    _inherit_punct_sizes,
    _is_punct_run,
    _is_stat_value,
    _should_apply_run_font_sizes,
    _snap_to_ladder,
    _unify_run_sizes_by_color,
    font_size_pt,
)
from layout.text_style import (  # noqa: E402, F401
    _glyph_height_in_bbox,
    detect_text_style,
)
from layout.text_units import (  # noqa: E402, F401
    _char_boxes_from_words,
    _char_width_unit,
    _derive_source_char_boxes,
    _estimated_runs_width_px,
    _estimated_text_width_px,
    _split_box_by_text_units,
    _text_width_units,
)
from layout.zorder import topo_sort_by_containment  # noqa: E402, F401


ENABLE_NATIVE_OUTLINE_SHAPES = False

# Emit confidently-fitted connector strokes as native PPT line elements
# (editable endpoints/colour/width) instead of raster PNGs. Crops the
# geometry fit can't explain keep the PNG fallback either way.
ENABLE_NATIVE_CONNECTORS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert inventory to manifest + layout.")
    parser.add_argument("--inventory", required=True,
                        help="Inventory JSON path.")
    parser.add_argument("--source", required=True,
                        help="Original source image (for style detection).")
    parser.add_argument("--cleaned", required=True,
                        help="Cleaned image (for asset crops).")
    parser.add_argument("--asset-prefix", required=True,
                        help="Asset path prefix in layout "
                             "(e.g. 'assets/page_001').")
    parser.add_argument("--out-assets-dir", required=True,
                        help="Directory to save asset PNGs.")
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-layout", required=True)
    parser.add_argument(
        "--slide-width-in", type=float, default=None,
        help="Slide width in inches. Default: auto-compute from "
             "source aspect ratio so 4:3 inputs get a 4:3 slide and "
             "16:9 inputs get 13.333 × 7.5.",
    )
    parser.add_argument("--slide-height-in", type=float, default=7.5)
    return parser.parse_args()


def run(*, inventory: str, source: str, cleaned: str,
        asset_prefix: str, out_assets_dir: str,
        out_manifest: str, out_layout: str,
        slide_width_in: float | None = None,
        slide_height_in: float = 7.5) -> None:
    """Programmatic entry — see parse_args() for the CLI equivalent.

    Used by run_pipeline.process_page to skip subprocess overhead.
    """
    args = argparse.Namespace(
        inventory=inventory, source=source, cleaned=cleaned,
        asset_prefix=asset_prefix, out_assets_dir=out_assets_dir,
        out_manifest=out_manifest, out_layout=out_layout,
        slide_width_in=slide_width_in, slide_height_in=slide_height_in,
    )
    _run(args)


def main() -> None:
    _run(parse_args())


def _run(args: argparse.Namespace) -> None:
    from layout.builder import LayoutBuilder
    builder = LayoutBuilder(args)
    builder.build()
    builder.write()


if __name__ == "__main__":
    main()
