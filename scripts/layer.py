import json

import gradio as gr

from backend.utils import load_torch_file
from modules.script_callbacks import on_ui_tabs

INFO: dict[str, str] = {}


def _load_inner(path: str) -> tuple[dict[str, str], bool]:
    INFO.clear()
    sd, meta = load_torch_file(path, return_metadata=True)

    keys = list(sd.keys())
    for k in keys:
        w = sd.pop(k)
        if k.endswith("comfy_quant"):
            INFO[k] = json.loads(w.numpy().tobytes())
        elif w.ndim == 0:
            INFO[k] = f"scaler ({w.dtype})"
        else:
            INFO[k] = f"{list(w.shape)} ({w.dtype})"

    return [
        gr.update(value=INFO, visible=True),
        gr.update(value=None, visible=True),
        gr.update(value=meta, visible=bool(meta)),
    ]


def load(path: str):
    try:
        return _load_inner(path)
    except Exception as e:
        gr.Warning(str(e))
        return [gr.update(value=None, visible=False)] * 3


def on_filter(search: str):
    return gr.update(value={k: v for k, v in INFO.items() if search in k})


def debug_ui():
    with gr.Blocks() as LAYER_DEBUG:
        with gr.Row():
            path = gr.Textbox(lines=1, max_lines=1, label="Path to Model", scale=9)
            btn = gr.Button("Load", variant="primary", scale=1)

        search = gr.Textbox(lines=1, max_lines=1, label="Filter", visible=False)
        info = gr.JSON(value=None, label="Layer Info", visible=False)
        meta = gr.JSON(value=None, label="Metadata", visible=False)

        for comp in (path, btn, search, info):
            comp.do_not_save_to_config = True

        btn.click(fn=load, inputs=[path], outputs=[info, search, meta])
        search.blur(fn=on_filter, inputs=[search], outputs=[info])

    return [(LAYER_DEBUG, "Layer Debug", "sd-forge-layer-debug")]


on_ui_tabs(debug_ui)
