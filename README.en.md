<h1 align="center">
  <img src="assets/logo.png" width="280" alt="Createdate">
</h1>

<p align="center">
  <strong>Createdate · Synthetic Multimodal Data for WeChat-Style Chats</strong>
</p>

<p align="center">
  Generate Chinese private or group chats, WeChat-style screenshots, and paired schedule labels with one command. Export datasets in LLaMA-Factory's multimodal ShareGPT format.
</p>

<p align="center">
  Try the examples without an API key · DeepSeek and compatible APIs · MIT licensed
</p>

<p align="center">
  <a href="README.md">简体中文</a> · <strong>English</strong>
</p>

<p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-3DA639" alt="Code License: MIT"></a>
</p>

<p align="center">
  <a href="#demo">Demo</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#generate-your-own-data">Generate Data</a> ·
  <a href="docs/architecture.md">Architecture (中文)</a> ·
  <a href="docs/usage.md">Usage Guide (中文)</a> ·
  <a href="#license">MIT License</a>
</p>

For developers and researchers building **schedule extraction datasets from Chinese chat screenshots**.
Createdate connects conversation generation, screenshot rendering, and paired data export to prepare samples for multimodal fine-tuning.
This repository includes the tools and two fictional examples; the author's working datasets are not distributed.

## Demo

<div align="center">
  <table align="center">
    <tr>
      <th align="center">Private chat · Picking up an art book</th>
      <th align="center">Group chat · Preparing a book club poster</th>
    </tr>
    <tr>
      <td align="center"><img src="docs/images/private_demo.png" width="280" alt="Fictional private chat arranging an art book pickup at a bookstore"></td>
      <td align="center"><img src="docs/images/group_demo.png" width="280" alt="Fictional group chat arranging to put up a poster at a bookstore"></td>
    </tr>
  </table>
  <p>Rendered with Createdate. Both conversations are <strong>fictional and written with AI assistance</strong>.</p>
</div>

The private chat on the left has this target label. The title means “pick up the art book at the bookstore”; labels remain in Chinese:

```json
{
  "items": [
    {
      "title": "去书店取画册",
      "date": "2026-09-21",
      "time": "15:00",
      "note": ""
    }
  ]
}
```

## Quick Start

Requires **Python 3.11+**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a local font with Chinese glyph support.
For an existing checkout, start at `uv sync --locked`.

```bash
# Clone the repository
git clone https://github.com/Ljw030710/wechat-dataset-generator.git
cd wechat-dataset-generator

# Install locked dependencies
uv sync --locked

# Read the example, render screenshots, and export paired data
uv run python scripts/run_pipeline.py private \
  --source examples/private_demo.json -o output/demo
```

| Output | Contents |
| --- | --- |
| `output/demo/images/private_demo.png` | Rendered private chat |
| `output/demo/llamafactory/images/` | Images bundled with the exported dataset |
| `output/demo/llamafactory/*_train.json`, `*_eval.json` | Training and evaluation records |
| `output/demo/llamafactory/dataset_info.json` | LLaMA-Factory field mappings and dataset registration |

**No API key or model service is needed for this example.** Installing dependencies for the first time requires internet access.
With the default split, the single example goes into the evaluation set and the training set is empty. This demonstrates the pipeline; training requires enough samples.

<details>
<summary>Try the group chat pipeline</summary>

```bash
uv run python scripts/run_pipeline.py group \
  --source examples/group_demo.json -o output/demo/group
```

The screenshot is saved to `output/demo/group/images/group_demo.png` and paired data to
`output/demo/group/llamafactory/`. No API key is needed.

</details>

<details>
<summary>If a Chinese font cannot be found</summary>

Pass an installed Chinese `.ttf` or `.ttc` font using `--font`. Replace the placeholder with a real local path:

```bash
uv run python scripts/run_pipeline.py private \
  --source examples/private_demo.json --font /path/to/chinese-font.ttf -o output/demo
```

Fonts are not bundled. Text layout may vary slightly across fonts and operating systems.

</details>

## Why Createdate?

- **Paired images and labels:** Render private or group chats and extract schedule labels from the same conversation. Outputs include `900 × 1949` single-screen PNGs by default, ShareGPT records, and `dataset_info.json`.
- **One complete pipeline:** Generate, render, and export with one command, or run each step separately. Conversations are saved individually so generation can resume from the existing count.
- **Quality checks:** Validate participants, dates, and times, then check labels against dialogue evidence without exposing the creative background. Failures trigger repair attempts; human review is still needed.

## Generate Your Own Data

Copy [.env.example](.env.example) to `.env` in the project root, then enter your provider's credentials and model settings:

```bash
cp -n .env.example .env
```

For DeepSeek, set these values in `.env` (replace the placeholder with your own key):

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_MODEL=deepseek-chat
```

The scripts load this file automatically. Then run:

```bash
# Generate → render → export
uv run python scripts/run_pipeline.py private \
  -n 1 --direction "朋友之间的日常协作" -o output/generated/private
```

The direction above means “everyday coordination between friends.” Outputs are Chinese conversations.
Results are saved as `conversations.json`, `images/`, and `llamafactory/` inside the selected directory.
Replace `private` with `group` for group chats. `-n` is the **target total**, including conversations already saved in that directory.
Rerunning reuses saved conversations, renders screenshots again, and exports the dataset. Use a new output directory for a fresh batch.

| Operation | Model calls |
| --- | --- |
| Rendering existing JSON and exporting with `--label-mode schedule` | None |
| Generating new conversations | Uses the selected provider |
| Exporting with `--label-mode teacher` | Sends source conversations to the selected provider for annotation |

A successful new conversation normally requires five model calls; repairs and retries add calls and may incur API costs.
To use CLIProxyAPI, set `LLM_PROVIDER=cliproxy` and its settings in `.env`. A local proxy may forward requests to an external model service.
For another provider, set `LLM_PROVIDER=custom`, `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
The endpoint must support Chat Completions and JSON object output. See the [configuration guide (Chinese)](docs/usage.md#自定义模型服务).

For all options, run `uv run python scripts/run_pipeline.py --help`; argument descriptions and runtime messages are currently in Chinese.
The [usage guide (Chinese)](docs/usage.md#一条命令完成全流程) also covers individual steps and resuming generation.

## Architecture and Documentation

Generation uses **Prompt Chaining**: scene brief → message beats → draft → editorial review,
followed by label evidence checks. Recent topics and dialogue endings are checked to reduce repetition.
See the [architecture document (Chinese)](docs/architecture.md) for implementation details and limitations.

| Resource | Contents |
| --- | --- |
| [Usage guide (Chinese)](docs/usage.md) | Provider configuration, JSON schemas, and commands |
| [Architecture (Chinese)](docs/architecture.md) | Prompt chaining, validation, repair, and image-label pairing |
| [Scripts](scripts/) | Generation, rendering, annotation, and export |
| [Examples](examples/) | Runnable fictional private and group conversations |
| [Tests](tests/) | Local automated tests |

## Scope and Limitations

- **Extraction target:** The first participant, `participants[0]`, is the default sender of the right-side green bubbles. New samples require this person to explicitly participate in one future agreement, relative to the conversation time.
- **Task coverage:** The current generator focuses on one explicit schedule item. No-event examples, multiple events, and ambiguous times require additional data construction.
- **Format versus performance:** Exports use LLaMA-Factory's multimodal ShareGPT format. Training performance and real-world accuracy have not been benchmarked; training must be configured separately in LLaMA-Factory.
- **Review:** Model-based evidence checks can make mistakes. Existing data reused on resume or through `--source`, and local `schedule` exports, do not automatically undergo a new model audit. Review synthetic conversations and labels, and add negative examples and independent evaluation data as appropriate.
- **Local data:** `data/`, `output/`, `archive/`, `llamafactory_*/`, and `work/` are ignored by Git and excluded from source releases.
- **Synthetic content:** The tool does not collect WeChat accounts or chat histories. Screenshots do not currently receive an automatic synthetic watermark; keep a clear fictional-content notice when sharing them.

## Contributing

Issues and pull requests are welcome for schedule edge cases, rendering, data quality checks, and reproducible training or evaluation examples.
For bug reports, include your operating system, Python version, command, and a minimal fictional JSON sample.

Local tests do not require a model API key:

```bash
uv run pytest -q
```

## License

Code, documentation, and fictional examples in `examples/` are available under the [MIT License](LICENSE).
Retain the copyright and license notice when using or redistributing them.

This is an independent project with no affiliation with or endorsement by Tencent or WeChat.
Third-party dependencies and model services have their own licenses or terms.
