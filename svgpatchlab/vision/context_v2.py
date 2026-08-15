from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from svgpatchlab.core.patch import extract_json_object
from svgpatchlab.models import ModelAdapter
from svgpatchlab.types import ModelRequest
from svgpatchlab.vision.context import VisionContextAnnotator, PROMPT_VERSION, DEFAULT_TAGS
from svgpatchlab.vision.render import render_node_comparison


# We increment PROMPT_VERSION so the cache doesn't mix with old generic zero-shot runs.
PROMPT_VERSION_V2 = 2


class VisionContextAnnotatorV2(VisionContextAnnotator):
    """
    Upgraded annotator that uses the dense, rule-based zero-shot prompt to 
    pack spatial and state data directly into the summary string for the final LLM.
    """
    def _cache_path(self, svg_hash: str, node_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        name = f"{node_id}-s{self.image_size}-v{PROMPT_VERSION_V2}.json"
        return self.cache_dir / svg_hash[:16] / name

    def _describe_node(
        self,
        node: dict[str, Any],
        rendered,
        model: ModelAdapter,
        request_id: str,
    ) -> dict[str, Any]:
        prompt = _vision_prompt_v2(node)
        response = model.generate(
            ModelRequest(
                prompt=prompt,
                images=(rendered.data_url,),
                metadata={"request_id": request_id},
            )
        )
        try:
            parsed = extract_json_object(response.text)
        except Exception:
            parsed = {"summary": response.text.strip(), "labels": []}
        return self._normalize_annotation(parsed)


def _vision_prompt_v2(node: dict[str, Any]) -> str:
    node_json = json.dumps(node, sort_keys=True)
    return f"""You are annotating one SVG DOM node for an SVG editing system.
The attached image has three panels:
1. left: the full SVG render
2. middle: only the selected node or node subtree
3. right: the full SVG render with the selected node highlighted in magenta

Compare the isolated node with the full image and describe what visible object
or object-part this node represents in context. 

Follow these strict rules for the "summary" field:
1. Identify the exact object or part (e.g., "tire", "window").
2. Include its spatial location relative to the whole image AND relative to important nearby nodes (e.g., "bottom-left tire attached to the red car").
3. Include its current visual state like color or pattern (e.g., "solid black").
4. If the node is only a tiny fragment, highlight, or shadow, state that clearly so it isn't confused with the main object.

For the "labels" array, use exact semantic terms a user might say in an edit instruction, such as window, eye, roof, shadow, outline, handle, letter, background, wheel, or highlight.

SVG node metadata:
{node_json}

Return only JSON with this shape:
{{"summary":"[color/state] [object/part] located at [relative position]","labels":["label1","label2"],"role":"object|part|style|shadow|outline|background|unknown","confidence":0.0}}
"""
