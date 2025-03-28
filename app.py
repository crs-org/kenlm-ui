"""
https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py
"""

import os
import gzip
import io
import sys
import subprocess
import functools

from importlib.metadata import version
from collections import Counter
from pathlib import Path

import gradio as gr

try:
    import kenlm
except ImportError:
    print("Please install `kenlm` library.")

# Config
title = "KenLM UI"

app_dir = "/home/hf-space/app"
kenlm_bin = f"{app_dir}/kenlm/build/bin"

examples = [
    ["demo.txt", 3, True],
]

description_head = f"""
# {title}

## Overview

This app gives you ability to debug KenLM models, enhance text using a trained model, and create a new KenLM model (Kneser-Ney) from a text corpus.
""".strip()


tech_env = f"""
#### Environment

- Python: {sys.version}
""".strip()

tech_libraries = f"""
#### Libraries

- kenlm: {version("kenlm")}
- gradio: {version("gradio")}
""".strip()


def convert_and_filter_topk(input_txt, top_k):
    """Convert to lowercase, count word occurrences and save top-k words to a file"""

    counter = Counter()
    data_lower = "/tmp/lower.txt.gz"

    print("\nConverting to lowercase and counting word occurrences ...")
    with io.TextIOWrapper(
        io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
    ) as file_out:
        # Open the input file either from input.txt or input.txt.gz
        _, file_extension = os.path.splitext(input_txt)
        if file_extension == ".gz":
            file_in = io.TextIOWrapper(
                io.BufferedReader(gzip.open(input_txt)), encoding="utf-8"
            )
        else:
            file_in = open(input_txt, encoding="utf-8")

        for line in file_in:
            line_lower = line.lower()
            counter.update(line_lower.split())
            file_out.write(line_lower)

        file_in.close()

    # Save top-k words
    print("\nSaving top {} words ...".format(top_k))
    top_counter = counter.most_common(top_k)
    vocab_str = "\n".join(word for word, count in top_counter)
    vocab_path = "/tmp/vocab-{}.txt".format(top_k)
    with open(vocab_path, "w+") as file:
        file.write(vocab_str)

    print("\nCalculating word statistics ...")
    total_words = sum(counter.values())
    print("  Your text file has {} words in total".format(total_words))
    print("  It has {} unique words".format(len(counter)))
    top_words_sum = sum(count for word, count in top_counter)
    word_fraction = (top_words_sum / total_words) * 100
    print(
        "  Your top-{} words are {:.4f} percent of all words".format(
            top_k, word_fraction
        )
    )
    print('  Your most common word "{}" occurred {} times'.format(*top_counter[0]))
    last_word, last_count = top_counter[-1]
    print(
        '  The least common word in your top-k is "{}" with {} times'.format(
            last_word, last_count
        )
    )
    for i, (w, c) in enumerate(reversed(top_counter)):
        if c > last_count:
            print(
                '  The first word with {} occurrences is "{}" at place {}'.format(
                    c, w, len(top_counter) - 1 - i
                )
            )
            break

    return data_lower, vocab_str


def inference_model(kenlm_model, text):
    if not kenlm_model:
        raise gr.Error("Please upload your KenLM model.")

    if not text:
        raise gr.Error("Please paste the text to score.")

    model = kenlm.Model(kenlm_model)
    results = []

    score = model.score(text, bos=True, eos=True)

    results.append(f"Score: {score}")
    results.append("---")

    # Show scores and n-gram matches
    words = ["<s>"] + text.split() + ["</s>"]
    for i, (prob, length, oov) in enumerate(model.full_scores(text)):
        results.append(
            "{0} {1}: {2}".format(prob, length, " ".join(words[i + 2 - length : i + 2]))
        )
        if oov:
            results.append('\t"{0}" is an OOV'.format(words[i + 1]))

    results.append("---")

    # Find out-of-vocabulary words
    for w in words:
        if w not in model:
            results.append('"{0}" is an OOV'.format(w))

    return "\n".join(results)


def score(lm, word, context):
    new_context = kenlm.State()
    full_score = lm.BaseFullScore(context, word, new_context)
    if full_score.oov:
        return -42, new_context  # odefault ov score looks too high
    return full_score.log_prob, new_context


@functools.lru_cache(maxsize=2**10)
def segment(lm, text, context=None, maxlen=20):
    if context is None:
        context = kenlm.State()
        lm.NullContextWrite(context)

    if not text:
        return 0.0, []

    textlen = min(len(text), maxlen)
    splits = [(text[: i + 1], text[i + 1 :]) for i in range(textlen)]

    candidates = []
    for word, remain_word in splits:
        first_prob, new_context = score(lm, word, context)
        remain_prob, remain_word = segment(lm, remain_word, new_context)

        candidates.append((first_prob + remain_prob, [word] + remain_word))

    return max(candidates)


def enhance_text(kenlm_model, text):
    if not kenlm_model:
        raise gr.Error("Please upload your KenLM model.")

    if not text:
        raise gr.Error("Please paste the text to score.")

    lm = kenlm.LanguageModel(kenlm_model)

    label = text.replace(" ", "")
    _, fixed_label_chunks = segment(lm, label)
    fixed_label = " ".join(fixed_label_chunks)

    return fixed_label


def generate_files(results):
    # Write words to a file
    words = [r.split() for r in results]
    words = list(set([w for r in words for w in r]))

    with open("/tmp/model_vocab.txt", "w") as f:
        f.write("\n".join(words))

    # Generate tokens file
    tokens = set()
    for word in words:
        tokens.update(list(word))
    # add "|" token
    tokens.add("|")

    with open("/tmp/model_tokens.txt", "w") as f:
        tokens_ordered = sorted(tokens)
        f.write("\n".join(tokens_ordered))

    # Generate lexicon file
    with open("/tmp/model_lexicon.txt", "w") as f:
        for word in words:
            splitted_word = " ".join(list(word + "|"))
            f.write(f"{word}\t{splitted_word}\n")


def text_to_kenlm(
    _text_file,
    _order,
    _do_lowercase,
    _binary_a_bits,
    _binary_b_bits,
    _binary_q_bits,
    _binary_type,
    _arpa_prune,
    _do_quantize,
    _topk_words,
    _do_limit_topk,
):
    if not _text_file:
        raise gr.Error("Please add a file.")

    if not _order:
        raise gr.Error("Please add an order.")

    gr.Info("Started to create a model, wait...")

    results = []

    # Read the file
    with open(_text_file, "r") as f:
        text = f.read()
        for line in text.split("\n"):
            if _do_lowercase:
                line = line.lower()
            results.append(line)

    # Remove previous files
    for file in [
        "/tmp/intermediate.txt", "/tmp/my_model.arpa", "/tmp/my_model-trie.bin", "/tmp/my_model_correct.arpa",
        "/tmp/my_model-trie-10000-words.arpa", "/tmp/my_model-trie-10000-words.bin",
        "/tmp/model_vocab.txt", "/tmp/model_lexicon.txt", "/tmp/model_tokens.txt",
    ]:
        if os.path.exists(file):
            os.remove(file)

    # Generate files: vocab, lexicon, tokens
    generate_files(results)

    # Write to intermediate file
    intermediate_file = "/tmp/intermediate.txt"
    with open(intermediate_file, "w") as f:
        f.write(" ".join(results))

    file_name = "/tmp/my_model.arpa"
    _do_model = True

    # Commands to run in the container
    if _do_model:
        cmd = (
            f"{kenlm_bin}/lmplz -T /tmp -S 80% --text {intermediate_file} --arpa /tmp/my_model.arpa -o {_order} --prune {_arpa_prune} --discount_fallback",
        )
        r = subprocess.run(cmd, shell=True)
        print(r)
        if r.returncode != 0:
            raise gr.Error("Failed to create model")

        file_name_fixed = "/tmp/my_model_correct.arpa"

        # Fix the ARPA file
        with (
            open(file_name, "r") as read_file,
            open(file_name_fixed, "w") as write_file,
        ):
            has_added_eos = False
            for line in read_file:
                if not has_added_eos and "ngram 1=" in line:
                    count = line.strip().split("=")[-1]
                    write_file.write(line.replace(f"{count}", f"{int(count) + 1}"))
                elif not has_added_eos and "<s>" in line:
                    write_file.write(line)
                    write_file.write(line.replace("<s>", "</s>"))
                    has_added_eos = True
                else:
                    write_file.write(line)
        # Replace the file name
        file_name = file_name_fixed

    if _do_limit_topk:
        file_name_words = f"/tmp/my_model-{_topk_words}-words.arpa"

        _, vocab_str = convert_and_filter_topk(intermediate_file, _topk_words)

        r = subprocess.run(
            [
                os.path.join(kenlm_bin, "filter"),
                "single",
                "model:{}".format(file_name),
                file_name_words,
            ],
            input=vocab_str.encode("utf-8"),
            check=True,
        )
        print(r)
        if r.returncode != 0:
            raise gr.Error("Failed to filter the model.")

        # Regenerate files: vocab, lexicon, tokens
        generate_files(vocab_str.split("\n"))

        if _do_quantize:
            file_name_quantized = (
                f"/tmp/my_model-{_binary_type}-{_topk_words}-words.bin"
            )

            cmd = f"{kenlm_bin}/build_binary -a {_binary_a_bits} -b {_binary_b_bits} -q {_binary_q_bits} -v {_binary_type} {file_name} {file_name_quantized}"
            r = subprocess.run(cmd, shell=True)
            print(r)
            if r.returncode != 0:
                raise gr.Error("Failed to quantize model")

            file_name = file_name_quantized
    else:
        if _do_quantize:
            file_name = f"/tmp/my_model-{_binary_type}.bin"

            cmd = f"{kenlm_bin}/build_binary -a {_binary_a_bits} -b {_binary_b_bits} -q {_binary_q_bits} -v {_binary_type} {file_name} {file_name}"
            r = subprocess.run(cmd, shell=True)
            print(r)
            if r.returncode != 0:
                raise gr.Error("Failed to quantize model")

    gr.Success("Model created.")

    model_file = gr.DownloadButton(
        value=Path(file_name), label=f"Download: {file_name}"
    )

    vocab_file = gr.DownloadButton(
        value=Path("/tmp/model_vocab.txt"),
        label="Created model_vocab.txt",
    )

    lexicon_file = gr.DownloadButton(
        value=Path("/tmp/model_lexicon.txt"),
        label="Created model_lexicon.txt",
    )

    tokens_file = gr.DownloadButton(
        value=Path("/tmp/model_tokens.txt"),
        label="Created model_tokens.txt",
    )

    return [model_file, vocab_file, lexicon_file, tokens_file]


with gr.Blocks(
    title=title,
    analytics_enabled=False,
    theme=gr.themes.Base(),
) as demo:
    gr.Markdown(description_head)
    gr.Markdown("## Usage")

    with gr.Tab("Evaluate"):
        with gr.Row():
            with gr.Column():
                kenlm_model = gr.File(label="KenLM model")

                text = gr.Text(label="Paste text")

            results = gr.Textbox(
                label="Scores",
                placeholder="Scores will be here.",
                show_copy_button=True,
                lines=10,
            )

        gr.Button("Run").click(
            inference_model,
            inputs=[kenlm_model, text],
            outputs=results,
        )

    with gr.Tab("Enhance"):
        with gr.Row():
            with gr.Column():
                kenlm_model = gr.File(label="Your KenLM model")

                text = gr.Text(label="Paste text to enhance")

            results = gr.Textbox(
                label="Results",
                placeholder="Results will be here.",
                show_copy_button=True,
                lines=10,
            )

        gr.Button("Run").click(
            enhance_text,
            inputs=[kenlm_model, text],
            outputs=results,
        )

    with gr.Tab("Create KenLM model"):
        with gr.Row():
            with gr.Column():
                text_file = gr.File(label="Text corpus")

                order = gr.Number(label="Order", value=3, minimum=1, maximum=5)

                do_lowercase = gr.Checkbox(
                    label="Lowercase text",
                )

                arpa_prune = gr.Text(
                    label="Prune",
                    value="0 1 1",
                )

                binary_a_bits = gr.Number(
                    label="Binary A bits",
                    value=256,
                )

                binary_b_bits = gr.Number(
                    label="Binary B bits",
                    value=7,
                )

                binary_q_bits = gr.Number(
                    label="Binary Q bits",
                    value=8,
                )

                binary_type = gr.Text(
                    label="Build binary data structure type",
                    value="trie",
                )

                do_quantize = gr.Checkbox(
                    label="Quantize model",
                    value=False,
                )

                topk_words = gr.Number(
                    label="Top-K words",
                    value=10000,
                )

                do_limit_topk = gr.Checkbox(
                    label="Limit vocabulary by Top-K words",
                    value=False,
                )

            with gr.Column():
                kenlm_model = gr.DownloadButton(
                    label="Created KenLM model",
                )

                vocab_file = gr.DownloadButton(
                    label="Created model_vocab.txt",
                )

                lexicon_file = gr.DownloadButton(
                    label="Created model_lexicon.txt",
                )

                tokens_file = gr.DownloadButton(
                    label="Created model_tokens.txt",
                )

        gr.Button("Create").click(
            text_to_kenlm,
            inputs=[
                text_file,
                order,
                do_lowercase,
                binary_a_bits,
                binary_b_bits,
                binary_q_bits,
                binary_type,
                arpa_prune,
                do_quantize,
                topk_words,
                do_limit_topk,
            ],
            outputs=[kenlm_model, vocab_file, lexicon_file, tokens_file],
        )

        with gr.Row():
            gr.Examples(
                label="Choose an example",
                inputs=[text_file, order, do_lowercase, do_quantize],
                examples=examples,
            )

    gr.Markdown("### Gradio app uses:")
    gr.Markdown(tech_env)
    gr.Markdown(tech_libraries)

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=True)
