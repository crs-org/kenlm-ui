import sys
import re

from importlib.metadata import version

import gradio as gr

# Config
concurrency_limit = 5

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
- kenlm: {version("kenlm")}
""".strip()


def clean_value(x):
    s = (
        x.replace("’", "'")
        .strip()
        .lower()
        .replace(":", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace("?", " ")
        .replace("!", " ")
        .replace("–", " ")
        .replace("«", " ")
        .replace("»", " ")
        .replace("—", " ")
        .replace("…", " ")
        .replace("/", " ")
        .replace("\\", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("́", "")
        .replace('"', " ")
    )

    s = re.sub(r" +", " ", s)

    s = s.strip()

    # print(s)

    return s


def inference(file_name, _clear_punctuation, _show_chars, _batch_mode):
    if not file_name:
        raise gr.Error("Please paste your JSON file.")

    df = pl.read_ndjson(file_name)

    required_columns = [
        "filename",
        "inference_start",
        "inference_end",
        "inference_total",
        "duration",
        "reference",
        "prediction",
    ]
    required_columns_batch = [
        "inference_start",
        "inference_end",
        "inference_total",
        "filenames",
        "durations",
        "references",
        "predictions",
    ]

    inference_seconds = df["inference_total"].sum()

    if _batch_mode:
        if not all(col in df.columns for col in required_columns_batch):
            raise gr.Error(
                f"Please provide a JSONL file with the following columns: {required_columns_batch}"
            )

        duration_seconds = 0
        for durations in df["durations"]:
            duration_seconds += durations.sum()

        rtf = inference_seconds / duration_seconds

        references_batch = df["references"]
        predictions_batch = df["predictions"]

        predictions = []
        for prediction in predictions_batch:
            if _clear_punctuation:
                prediction = prediction.map_elements(
                    clean_value, return_dtype=pl.String
                )
                predictions.extend(prediction)
            else:
                predictions.extend(prediction)

        references = []
        for reference in references_batch:
            references.extend(reference)
    else:
        if not all(col in df.columns for col in required_columns):
            raise gr.Error(
                f"Please provide a JSONL file with the following columns: {required_columns}"
            )

        duration_seconds = df["duration"].sum()

        rtf = inference_seconds / duration_seconds

        references = df["reference"]

        if _clear_punctuation:
            predictions = df["prediction"].map_elements(
                clean_value, return_dtype=pl.String
            )
        else:
            predictions = df["prediction"]

    n_predictions = len(predictions)
    n_references = len(references)

    # Evaluate
    wer_value = round(wer.compute(predictions=predictions, references=references), 4)
    cer_value = round(cer.compute(predictions=predictions, references=references), 4)

    inference_time = inference_seconds
    audio_duration = duration_seconds

    rtf = inference_time / audio_duration

    results = []

    results.append(
        f"- Number of references / predictions: {n_references} / {n_predictions}"
    )
    results.append(f"")
    results.append(f"- WER: {wer_value} metric, {round(wer_value * 100, 4)}%")
    results.append(f"- CER: {cer_value} metric, {round(cer_value * 100, 4)}%")
    results.append("")
    results.append(f"- Accuracy on words: {round(100 - 100 * wer_value, 4)}%")
    results.append(f"- Accuracy on chars: {round(100 - 100 * cer_value, 4)}%")
    results.append("")
    results.append(
        f"- Inference time: {round(inference_time, 4)} seconds, {round(inference_time / 60, 4)} mins, {round(inference_time / 60 / 60, 4)} hours"
    )
    results.append(
        f"- Audio duration: {round(audio_duration, 4)} seconds, {round(audio_duration / 60 / 60, 4)} hours"
    )
    results.append("")
    results.append(f"- RTF: {round(rtf, 4)}")

    if _show_chars:
        all_chars = set()
        for pred in predictions:
            for c in pred:
                all_chars.add(c)

        sorted_chars = natsorted(list(all_chars))

        results.append("")
        results.append(f"Chars in predictions:")
        results.append(f"{sorted_chars}")

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
            jsonl_file = gr.File(label="A JSONL file")

            clear_punctuation = gr.Checkbox(
                label="Clear punctuation, some chars and convert to lowercase",
            )
            show_chars = gr.Checkbox(
                label="Show chars in predictions",
            )
            batch_mode = gr.Checkbox(
                label="Use batch mode",
            )

        metrics = gr.Textbox(
            label="Metrics",
            placeholder=metrics_value,
            show_copy_button=True,
        )

    gr.Button("Calculate").click(
        inference,
        concurrency_limit=concurrency_limit,
        inputs=[jsonl_file, clear_punctuation, show_chars, batch_mode],
        outputs=metrics,
    )

    with gr.Row():
        gr.Examples(
            label="Choose an example",
            inputs=[jsonl_file, clear_punctuation, show_chars, batch_mode],
            examples=examples,
        )

    gr.Markdown("### Gradio app uses:")
    gr.Markdown(tech_env)
    gr.Markdown(tech_libraries)

if __name__ == "__main__":
    demo.queue()
    demo.launch()
