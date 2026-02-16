# Audio Arena — Hard Static Benchmark

This is the **hard static benchmark** used by [Audio Arena](https://audioarena.ai), a project by [Arcada Labs](https://arcada.dev) for evaluating voice AI models.

- **Leaderboard & results**: [audioarena.ai](https://audioarena.ai)
- **Dataset on Hugging Face**: [arcada-labs/audio-arena](https://huggingface.co/datasets/arcada-labs/audio-arena)
- **Source code**: [github.com/Design-Arena/audio-arena](https://github.com/Design-Arena/audio-arena)

## Background

This benchmark builds on the original [30-turn multi-turn evaluation](https://github.com/kwindla/aiewf-eval) ([blog post](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/)), which tested text and speech-to-speech models on tool use, instruction following, and knowledge base grounding in an AI Engineer World's Fair conference assistant scenario.

The original 30 questions turned out to not be sufficiently challenging — most frontier models scored above 90% on nearly every category. We discarded the majority of the original turns and rebuilt the benchmark from scratch as a **75-turn static hard benchmark**. Only a small number of basic QA and tool-use turns from the original were retained, and even those were revised.

The new benchmark is 2.5x larger and substantially harder, with turns specifically designed to test adversarial traps, multi-step tool use, long-range memory, error recovery, cancellation flows, ambiguity handling, implicit correction, and distractor injection. See the [Methodology](#methodology) section below for the full details and scoring rubric.

The benchmark uses a ~12K token knowledge base with 75 turns, 9 tools, and pre-recorded audio files.

## Quick Start

```bash
# Install dependencies
uv sync

# List available benchmarks
uv run audio-arena list-benchmarks

# Run a benchmark. Results will be saved to runs/conference_assistant/<timestamp>_<model_name>
uv run audio-arena run conference_assistant --model claude-sonnet-4-5 --service anthropic

# Judge the results
uv run audio-arena judge runs/conference_assistant/<timestamp>_claude-sonnet-4-5
```

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Design-Arena/audio-arena.git
cd audio-arena
uv sync
```

## Environment Variables

Set the API keys for the services you want to use. You only need the keys for the models you plan to test.

```bash
# Required for judging (Claude evaluates all benchmark results)
export ANTHROPIC_API_KEY=sk-ant-...

# Model provider keys (set whichever you need)
export OPENAI_API_KEY=sk-...          # OpenAI (text and realtime)
export GOOGLE_API_KEY=...             # Google (Gemini text and Gemini Live)
export OPENROUTER_API_KEY=...         # OpenRouter (access to multiple providers)
export GROQ_API_KEY=...               # Groq
export CEREBRAS_API_KEY=...           # Cerebras
export ULTRAVOX_API_KEY=...           # Ultravox
export XAI_API_KEY=...                # xAI (Grok Realtime)

# AWS Nova models (text and speech-to-speech)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...          # Optional: for temporary credentials
export AWS_REGION=us-east-1           # Optional: defaults to us-east-1
```

You can also create a `.env` file in the project root with these variables.

## CLI Commands

### Running Benchmarks

```bash
# Basic usage with text model
uv run audio-arena run <benchmark> --model <model> --service <service>

# Examples:
uv run audio-arena run conference_assistant --model claude-sonnet-4-5 --service anthropic
uv run audio-arena run conference_assistant --model gpt-4o --service openai
uv run audio-arena run conference_assistant --model gemini-2.5-flash --service google

# Realtime audio models 
uv run audio-arena run conference_assistant --model gpt-realtime --service openai-realtime
uv run audio-arena run conference_assistant --model gemini-2.5-flash-native-audio-preview-12-2025 --service gemini-live
uv run audio-arena run conference_assistant --model ultravox-v0.7 --service ultravox-realtime

# Nova Sonic (no --service needed, pipeline creates its own LLM)
uv run audio-arena run conference_assistant --model amazon.nova-2-sonic-v1:0 --pipeline nova-sonic

# Grok (xAI) Realtime
uv run audio-arena run conference_assistant --model grok-realtime

# Debug with limited turns
uv run audio-arena run conference_assistant --model gpt-4o --service openai --only-turns 0,1,2

# Verbose logging
uv run audio-arena run conference_assistant --model gpt-4o --service openai --verbose
```

### Judging Runs

After a benchmark run completes, judge the results using Claude:

```bash
# Judge a specific run
uv run audio-arena judge runs/conference_assistant/20251213T123456_claude-sonnet-4-5

# Judge with specific turns
uv run audio-arena judge runs/conference_assistant/20251213T123456_claude-sonnet-4-5 --only-turns 0,1,2

# Use a different judge model
uv run audio-arena judge runs/conference_assistant/20251213T123456_claude-sonnet-4-5 --judge-model claude-sonnet-4-5
```

The Claude judge evaluates each turn on up to 5 dimensions:

| Dimension | Scored on | Description |
|-----------|-----------|-------------|
| `tool_use_correct` | Every turn | Did the model call the expected function with correct arguments? |
| `instruction_following` | Every turn | Did the model answer the question or advance the task? |
| `kb_grounding` | Every turn | Is the response factually consistent with the knowledge base? |
| `state_tracking` | Turns tagged with long-range memory, cancellation flow, or implicit correction | Does the model correctly track information from earlier turns? |
| `ambiguity_handling` | Turns tagged with ambiguous entity or compound ambiguity | Does the model correctly disambiguate entities and constraints? |

For speech-to-speech runs, a 6th dimension is added automatically when a `conversation.wav` file is present:

| Dimension | Description |
|-----------|-------------|
| `turn_taking` | Audio timing correctness — detects missing responses, negative TTFB, empty responses (control tokens only), alignment drift, and audio overlap |

When turn-taking failures occur, the judge is more lenient on `instruction_following` since garbled audio may cause transcription issues.

```bash
# Judge a speech-to-speech run (turn-taking analysis runs automatically)
uv run audio-arena judge runs/conference_assistant/20260111T123456_gpt-realtime_abc123

# Skip turn-taking analysis
uv run audio-arena judge runs/conference_assistant/20260111T123456_gpt-realtime_abc123 --skip-turn-taking
```

Judge outputs (saved to the run directory):
- `claude_summary.json` — score metrics (includes `turn_taking_failures` for S2S runs)
- `claude_analysis.md` — human-readable report with failures
- `claude_judged.jsonl` — per-turn judgments with reasoning

See the [Methodology](#methodology) section for details on two-phase evaluation, penalty absorption, and category-aware scoring.

### Listing Options

```bash
# List available benchmarks
uv run audio-arena list-benchmarks

# List available pipelines
uv run audio-arena list-pipelines

# List service aliases
uv run audio-arena list-aliases
```

## Service Aliases

For convenience, common service classes have short aliases:

| Alias | Provider |
|-------|----------|
| `openai` | OpenAI (text models) |
| `openai-realtime` | OpenAI Realtime (speech-to-speech) |
| `anthropic` | Anthropic (Claude models) |
| `google` | Google (Gemini text models) |
| `gemini-live` | Google Gemini Live (speech-to-speech) |
| `openrouter` | OpenRouter (multi-provider, OpenAI-compatible) |
| `groq` | Groq |
| `cerebras` | Cerebras |
| `bedrock` | AWS Bedrock (Nova text models) |
| `ultravox-realtime` | Ultravox (speech-to-speech) |

You can also use fully-qualified class names:

```bash
uv run audio-arena run conference_assistant \
    --model gpt-4o \
    --service pipecat.services.openai.llm.OpenAILLMService
```

## Benchmarks

Benchmarks are located in `benchmarks/`. Each benchmark is a self-contained Python package with:
- `config.py` - Benchmark configuration (turns, tools, system instruction)
- `turns.py` - Turn definitions with golden answers
- `tools.py` - Tool/function schema definitions
- `system.py` - System prompt with knowledge base
- `data/knowledge_base.txt` - Knowledge base content
- `audio/` - Pre-recorded audio files for each turn

### Available Benchmarks

| Benchmark | Description | Knowledge Base |
|-----------|-------------|----------------|
| `conference_assistant` | 75-turn hard benchmark | ~12K tokens |

## Pipelines

| Pipeline | Use Case | Auto-Detection Pattern |
|----------|----------|------------------------|
| `text` | Synchronous text LLMs | Default for all models |
| `realtime` | OpenAI Realtime, Gemini Live, Ultravox | `*realtime*`, `*native-audio*`, `*live*`, `*ultravox*` |
| `grok-realtime` | xAI Grok Realtime | `grok*realtime*` |
| `nova-sonic` | AWS Nova Sonic | `*nova-sonic*`, `*nova_sonic*` |

## Output Structure

Runs are saved to `runs/<benchmark>/<timestamp>_<model>/`:

```
runs/
└── conference_assistant/
    └── 20251213T123456_claude-sonnet-4-5/
        ├── transcript.jsonl        # Turn-by-turn results
        ├── runtime.json            # Run metadata and metrics
        ├── run.log                 # Debug logs
        ├── claude_summary.json     # Judge summary (after judging)
        ├── claude_judged.jsonl     # Per-turn judgments (after judging)
        └── claude_analysis.md      # Human-readable analysis (after judging)
```

## Project Structure

```
audio-arena/
├── src/audio_arena/               # Main package
│   ├── cli.py                     # CLI entry point
│   ├── pipelines/                 # Pipeline implementations
│   │   ├── base.py                # Abstract base pipeline
│   │   ├── text.py                # Text pipeline
│   │   ├── realtime.py            # Realtime pipeline (Gemini/Ultravox)
│   │   ├── openai_realtime.py     # OpenAI Realtime pipeline
│   │   ├── grok_realtime.py       # Grok Realtime pipeline
│   │   └── nova_sonic.py          # Nova Sonic pipeline
│   ├── processors/                # Frame processors
│   │   ├── tool_call_recorder.py  # Records tool calls
│   │   ├── tts_transcript.py      # TTS transcript handling
│   │   └── audio_buffer.py        # Audio buffer processing
│   ├── transports/                # Input/output transports
│   │   ├── paced_input.py         # Paced audio input
│   │   └── null_audio_output.py   # Null audio sink
│   ├── recording/                 # Transcript recording
│   │   └── transcript_recorder.py # Records transcripts
│   └── judging/                   # Judge implementations
│       ├── claude_judge.py        # Claude-based judging
│       └── turn_taking.py         # Turn-taking analysis
│
├── benchmarks/
│   └── conference_assistant/      # 75-turn hard benchmark
│       ├── config.py
│       ├── turns.py
│       ├── tools.py
│       ├── system.py
│       ├── audio/
│       └── data/
│
├── scripts/
│   └── analyze_turn_metrics.py    # Per-turn timing analysis
├── runs/                          # Output directory (gitignored)
├── LICENSE
└── pyproject.toml
```

## Comprehensive Turn Metrics Analysis

For detailed per-turn timing analysis of speech-to-speech models, use the comprehensive metrics script:

```bash
# Analyze a run with summary statistics
uv run python scripts/analyze_turn_metrics.py runs/conference_assistant/<timestamp>_<model>

# Show per-turn breakdown table
uv run python scripts/analyze_turn_metrics.py runs/conference_assistant/<timestamp>_<model> -v

# Output as JSON (for programmatic use)
uv run python scripts/analyze_turn_metrics.py runs/conference_assistant/<timestamp>_<model> --json
```

### Metrics Explained

The script consolidates timing data from multiple sources and calculates the following metrics:

| Metric | Description | Calculation |
|--------|-------------|-------------|
| **Server TTFB** | Time from request to first byte from model | Read from `transcript.jsonl` (reported by Pipecat) |
| **Pipeline TTFB** | Time from user speech end to bot audio tag | `bot_tag_log_ms - user_end_ms` (Silero VAD) |
| **WAV V2V** | Voice-to-voice latency measured from audio | `bot_silero_start_ms - user_end_ms` (Silero VAD) |
| **Silent Pad (RMS)** | Silent padding before speech (RMS detection) | `bot_rms_onset_ms - bot_tag_log_ms` |
| **Silent Pad (VAD)** | Silent padding before speech (Silero VAD) | `bot_silero_start_ms - bot_tag_wav_ms` |
| **Tag Alignment** | Drift between log position and WAV detection | `bot_tag_log_ms - bot_tag_wav_ms` |

**Key metric relationships:**
- **WAV V2V = Pipeline TTFB + Silent Pad (VAD)** - The total voice-to-voice latency includes both the time waiting for audio to arrive and any initial silence in the audio stream
- **Pipeline TTFB** measures when audio starts arriving at the pipeline
- **Silent Pad** measures how much silence is at the beginning of the audio (most models send 40-120ms of silence before speech)

### Alignment Sanity Check

The script verifies that log-based timestamps match actual audio positions by detecting audio tags (2kHz tones) embedded in the WAV file:

- **Bot tags**: Inserted when bot audio arrives at the pipeline
- **Alignment OK**: Log and WAV positions match within ±20ms tolerance
- **Issues detected**: Missing tags, extra tags, or drift outside tolerance

### Output Files

When run with `--json`, the script outputs structured data that can be saved:

```bash
# Save metrics to JSON file
uv run python scripts/analyze_turn_metrics.py runs/conference_assistant/<timestamp>_<model> --json > turn_metrics.json
```

## Methodology

The new and redesigned turns specifically test:

- **Adversarial traps** — Authority appeals, plausible hallucinations, subtle prompt injection, near-miss entities, and false recall
- **Multi-step tool use and long-range memory** — Conditional logic, parallel chains, implicit requirements, rollbacks, and correct use of information from many turns earlier
- **Error recovery** — Cascading failures, partial success states, and ambiguous error messages
- **Cancellation flow and state tracking** — User changes of mind and correct handling of cancelled actions across turns
- **Ambiguity handling** — Ambiguous entities (e.g., two people with the same name), compound ambiguity, and dependent or contradictory constraints
- **Implicit correction** — Nested misconceptions, partial truths, and false attributions that the model must correct without over-correcting
- **Distractor injection** — Buried questions, emotional manipulation, and technical tangents that require focusing on the actual user intent

### Scoring Rubric

- **Category-aware dimensions** — Core dimensions (tool use, instruction following, KB grounding) are scored on every turn. The dimensions `state_tracking` and `ambiguity_handling` are scored only on turns tagged with the relevant categories (e.g., long-range memory, cancellation flow, implicit correction, ambiguous entity), so we do not penalize models on dimensions that are out of scope for a given turn.
- **Two-phase evaluation** — An initial turn-by-turn pass is followed by a realignment pass that detects early or late function calls and cascading effects. If a required call was made a turn early, later turns that "expected" that call are not penalized for a "missing" call; if a call was made a turn late, the turn where it was actually made gets credit.
- **Penalty absorption** — When a missed tool call has a more specific root cause, the penalty lands on that dimension instead of `tool_use_correct`. If a model over-clarified (asked for confirmation when it wasn't needed) and `ambiguity_handling` is in scope, the penalty goes to `ambiguity_handling`. If a model forgot earlier conversational state and `state_tracking` is in scope, the penalty goes to `state_tracking`. This avoids double-penalizing while ensuring every failure is counted exactly once.
- **Strict separation of instruction_following and tool_use_correct** — Failing to call a tool when expected is scored only under `tool_use_correct` (or absorbed by a more specific dimension). `instruction_following` is failed only when the assistant's words and actions contradict each other in a non-tool sense.
- **Turn-taking and leniency** — For speech-to-speech runs, a pre-computed `turn_taking` dimension reflects audio timing (overlaps, interruptions, missing response). When turn_taking fails, the judge is more lenient on instruction_following to account for possible transcription or audio issues.

The benchmark is **static**: the same 75 user inputs (and corresponding audio) are used for every run, with golden expectations and category tags defined in `benchmarks/conference_assistant/turns.py`, so results are comparable across models and runs.

## Acknowledgments

Audio Arena is built on top of the [Pipecat](https://github.com/pipecat-ai/pipecat) open-source framework for voice and multimodal AI. The original 30-turn evaluation was created by [Kwindla Hultman Kramer](https://github.com/kwindla) at [Daily](https://www.daily.co/) — see the [original blog post](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/) and [original repo](https://github.com/kwindla/aiewf-eval).

Judging is powered by [Claude](https://www.anthropic.com/) via the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk). Voice activity detection uses [Silero VAD](https://github.com/snakers4/silero-vad).

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{audioarena2026,
  title={AudioArena: A Multi-Turn Speech-to-Speech Evaluation Benchmark},
  author={Arcada Labs},
  year={2026},
  url={https://huggingface.co/datasets/arcada-labs/audio-arena}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
