import sys
import re

from importlib.metadata import version

import gradio as gr

# Config
title = "KenLM training"

examples = [
    ["uk.txt", True],
    ["uk_ru.txt", True],
]

description_head = f"""
# {title}

## Overview

Upload a text file.
""".strip()


metrics_value = """
Commands will be here.
""".strip()

tech_env = f"""
#### Environment

- Python: {sys.version}
""".strip()

tech_libraries = f"""
#### Libraries

- gradio: {version("gradio")}
""".strip()


def inference(file_name, _do_lowercase):
    if not file_name:
        raise gr.Error("Please paste your JSON file.")

    results = []

    results.append(f"")

    return "\n".join(results)


demo = gr.Blocks(
    title=title,
    analytics_enabled=False,
    theme=gr.themes.Base(),
)

with demo:
    gr.Markdown(description_head)

    gr.Markdown("## Usage")

    with gr.Row():
        with gr.Column():
            jsonl_file = gr.File(label="A file")

            do_lowercase = gr.Checkbox(
                label="Lowercase text",
            )

        results = gr.Textbox(
            label="Metrics",
            placeholder=metrics_value,
            show_copy_button=True,
        )

    gr.Button("Perform").click(
        inference,
        inputs=[jsonl_file, do_lowercase],
        outputs=results,
    )

    with gr.Row():
        gr.Examples(
            label="Choose an example",
            inputs=[jsonl_file, do_lowercase],
            examples=examples,
        )

    gr.Markdown("### Gradio app uses:")
    gr.Markdown(tech_env)
    gr.Markdown(tech_libraries)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
