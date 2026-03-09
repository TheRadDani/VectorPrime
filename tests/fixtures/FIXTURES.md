# Test Fixtures

Model fixture files are large and not committed to git.
Download them manually before running fixture-dependent tests.

## GGUF Fixture (TinyLlama 1.1B Q4_K_M)

```bash
mkdir -p tests/fixtures
curl -L \
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  -o tests/fixtures/tiny.gguf
```

## ONNX Fixture (distilbert-base-uncased)

```bash
pip install optimum[exporters]
optimum-cli export onnx \
  --model distilbert-base-uncased \
  tests/fixtures/tiny_onnx/
```

## Running fixture-dependent tests

```bash
# After downloading fixtures:
pytest tests/ -v

# GPU tests (requires NVIDIA GPU):
LLMFORGE_GPU_TESTS=1 pytest tests/ -v -m requires_gpu
```

## Skipping fixture tests in CI

Tests marked `@pytest.mark.requires_fixtures` are automatically
skipped in CI when the fixture files are absent. This is the correct
behaviour — CI validates pure logic and the CLI smoke tests only.
