# Cerebellum Gemma 4 26B Server

This branch is a small llama.cpp runtime branch for serving Cerebellum Gemma 4
26B GGUFs with reasoning and vision enabled.

It keeps upstream llama.cpp history intact and only carries local server changes
needed to make Gemma 4 operation harder to misconfigure.

## Runtime Fixes

- Build `llama-server` with OpenSSL so HTTPS image URLs work for multimodal
  requests.
- Preserve upstream Gemma 4 reasoning-budget fixes, including the fix that
  prevents prompt tokens from consuming the reasoning budget.
- Let explicit request-level `thinking_budget_tokens` override a server default
  from `--reasoning-budget`. This prevents stale fixed-budget launchers from
  silently defeating client requests for short thinking or no thinking.

## Build

Install OpenSSL headers in the build environment, then configure and build:

```bash
sudo apt-get update
sudo apt-get install -y libssl-dev pkg-config

cmake -S . -B build -DLLAMA_OPENSSL=ON
cmake --build build --config Release --target llama-server -j 12
```

Confirm CMake found OpenSSL:

```bash
grep -E "LLAMA_OPENSSL|OPENSSL_.*LIBRARY|OPENSSL_INCLUDE_DIR" build/CMakeCache.txt
```

The output must point at real OpenSSL include and library paths, not
`*-NOTFOUND`.

## Launch

Use request-level budgets for apps and agents. Do not set a fixed server budget
unless every client should get the same budget.

```bash
build/bin/llama-server \
  --model /path/to/gemma-4-26B-A4B-it-heretic-cerebellum-v1.1-templatefix.gguf \
  --mmproj /path/to/gemma-4-26B-A4B-it-heretic.mmproj-f16.gguf \
  --host 127.0.0.1 \
  --port 7830 \
  --n-gpu-layers 99 \
  --ctx-size 131072 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --reasoning auto \
  --media-path /tmp/ \
  --alias gemma4-26b-cerebellum-templatefix-vision \
  --no-warmup
```

For published no-thinking benchmarks, keep using llama-server with reasoning
disabled at request time:

```json
{
  "chat_template_kwargs": { "enable_thinking": false },
  "thinking_budget_tokens": 0
}
```

For interactive thinking, send a finite request budget:

```json
{
  "thinking_budget_tokens": 128
}
```

## Smoke Tests

Text reasoning budget check:

```bash
curl -s http://127.0.0.1:7830/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-26b-cerebellum-templatefix-vision",
    "messages": [
      {"role": "user", "content": "Think briefly, then answer: what is 17 * 23?"}
    ],
    "max_tokens": 512,
    "thinking_budget_tokens": 128
  }'
```

Vision check with a local image:

```bash
curl -s http://127.0.0.1:7830/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-26b-cerebellum-templatefix-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image in one sentence."},
        {"type": "image_url", "image_url": {"url": "file://vision.png"}}
      ]
    }],
    "max_tokens": 512,
    "thinking_budget_tokens": 128
  }'
```

HTTPS vision check:

```bash
curl -s http://127.0.0.1:7830/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma4-26b-cerebellum-templatefix-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe the image in one sentence."},
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg"}}
      ]
    }],
    "max_tokens": 512,
    "thinking_budget_tokens": 128
  }'
```

If HTTPS support is missing, llama-server raises:

```text
HTTPS is not supported. Please rebuild with one of:
```

## Credits

- Runtime: ggml-org/llama.cpp, MIT license.
- Base model: Google Gemma team, Gemma 4 26B-A4B-it.
- Quantization method and release work: deucebucket/cerebellum.
- Calibration/imatrix provenance should be credited in the model card for each
  released GGUF when used.

Carry the llama.cpp license and third-party notices with any binary release.
For Gemma model files, follow the license and terms stored in the GGUF metadata
and linked by the model publisher.
