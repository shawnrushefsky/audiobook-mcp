# Adding New TTS Engines

Follow these steps to add a new TTS engine. Use existing engines as reference (e.g., `xtts.py` for audio-prompted, `maya1.py` for text-prompted).

## Step 1: Create the Engine File

Create `talky_talky/tools/tts/<engine_name>.py`:

```python
"""<EngineName> TTS Engine - Brief description."""

import sys
from pathlib import Path

from .base import AudioPromptedEngine, TTSResult, EngineInfo, PromptingGuide
# Or use TextPromptedEngine for voice-description-based engines
from .utils import split_text_into_chunks, get_best_device, get_available_memory_gb

# Constants
SAMPLE_RATE = 24000  # Output sample rate
MAX_CHUNK_CHARS = 400  # Max chars per generation chunk

# Lazy-loaded model singleton
_model = None

def _load_model():
    """Lazily load the model."""
    global _model
    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()
    print(f"Loading <EngineName> on {device}...", file=sys.stderr, flush=True)

    # Load model here
    # Handle device compatibility (CUDA, MPS, CPU)

    print("<EngineName> loaded successfully", file=sys.stderr, flush=True)
    return _model

class MyEngine(AudioPromptedEngine):  # or TextPromptedEngine
    @property
    def name(self) -> str:
        return "Engine Display Name"

    @property
    def engine_id(self) -> str:
        return "engine_id"  # lowercase, used in API

    def is_available(self) -> bool:
        """Check if dependencies are installed and device is compatible."""
        try:
            import required_package  # noqa: F401
            # Add device checks if needed (e.g., CUDA-only)
            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",  # or "text_prompted"
            description="Brief description",
            requirements="package-name (pip install package-name)",
            max_duration_secs=30,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,  # "[tag]" or "<tag>"
            emotion_tags=[],
            extra_info={...},
            prompting_guide=PromptingGuide(...),  # Optional but recommended
        )

    def get_setup_instructions(self) -> str:
        return """## Engine Setup Instructions..."""

    def generate(self, text, output_path, reference_audio_paths, **kwargs) -> TTSResult:
        """Generate audio. Always return TTSResult."""
        import soundfile as sf

        output_path = Path(output_path)

        # Validate inputs
        if not text.strip():
            return TTSResult(status="error", output_path=str(output_path),
                           duration_ms=0, sample_rate=SAMPLE_RATE, error="Empty text")

        try:
            model = _load_model()
            # Generate audio...
            audio = model.generate(text, ...)

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, SAMPLE_RATE)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=int(len(audio) / SAMPLE_RATE * 1000),
                sample_rate=SAMPLE_RATE,
                metadata={...},
            )
        except Exception as e:
            return TTSResult(status="error", output_path=str(output_path),
                           duration_ms=0, sample_rate=SAMPLE_RATE, error=str(e))
```

## Step 2: Register the Engine

In `talky_talky/tools/tts/__init__.py`:

```python
# Add import at top with other engines
from .myengine import MyEngine

# Add to register_engine calls
register_engine(MyEngine)

# Add to __all__
__all__ = [..., "MyEngine"]
```

## Step 3: Add the MCP Tool

In `talky_talky/server/app.py`:

```python
@mcp.tool()
def speak_myengine(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],  # For audio-prompted
    # voice_description: str,  # For text-prompted
    # Add engine-specific params with defaults
) -> dict:
    """Generate speech using MyEngine.

    Detailed docstring for AI agents...
    """
    result = generate(
        text=text,
        output_path=output_path,
        engine="myengine",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)
```

## Step 4: Add Dependencies

In `pyproject.toml`:

```toml
[project.optional-dependencies]
# Add new engine extra
myengine = [
    "required-package>=1.0.0",
]
# Update tts to include it
tts = [
    "talky-talky[maya1,chatterbox,mira,xtts,myengine]",
]
```

Then update the lock file:
```bash
uv lock
```

## Step 5: Handle Device Compatibility

Important considerations:

1. **CUDA-only engines** (like MiraTTS): Check `torch.cuda.is_available()` in `is_available()`
2. **MPS support**: Use `get_best_device()` and handle MPS-specific loading
3. **PyTorch 2.6+ compatibility**: If loading pickled weights, patch `torch.load`:
   ```python
   import torch
   from functools import wraps

   _original = torch.load
   @wraps(_original)
   def _patched(*args, **kwargs):
       kwargs.setdefault("weights_only", False)
       return _original(*args, **kwargs)
   torch.load = _patched
   try:
       # Load model
   finally:
       torch.load = _original
   ```

## Verification Steps

After implementing, run these verification steps:

### 1. Check Engine Registration

```bash
uv run python -c "
from talky_talky.tools.tts import list_engines, get_engine

engines = list_engines()
print('Registered engines:', list(engines.keys()))

engine = get_engine('myengine')
print(f'Name: {engine.name}')
print(f'Available: {engine.is_available()}')
"
```

### 2. Install Dependencies

```bash
uv pip install -e ".[myengine]"
```

### 3. Test Generation

```bash
uv run python -c "
from talky_talky.tools.tts import generate

result = generate(
    text='Hello, this is a test.',
    output_path='/tmp/test_output.wav',
    engine='myengine',
    reference_audio_paths=['/path/to/reference.wav'],  # if audio-prompted
)
print(f'Status: {result.status}')
print(f'Duration: {result.duration_ms}ms')
if result.error:
    print(f'Error: {result.error}')
"
```

### 4. Verify Audio Output

```bash
# Check file exists and has content
ls -la /tmp/test_output.wav

# Get audio info
uv run python -c "
from talky_talky.tools.audio import get_audio_info
info = get_audio_info('/tmp/test_output.wav')
print(f'Duration: {info.duration_ms}ms')
print(f'Format: {info.format}')
"
```

### 5. Run Linter

```bash
uvx ruff check talky_talky/tools/tts/myengine.py
uvx ruff format talky_talky/tools/tts/myengine.py
```

### 6. Test MCP Tool

```bash
uv run python -c "
from talky_talky.server import speak_myengine

result = speak_myengine(
    text='Testing the MCP tool.',
    output_path='/tmp/mcp_test.wav',
    reference_audio_paths=['/path/to/reference.wav'],
)
print(result)
"
```

### 7. Update Documentation

- Update `README.md` with engine description and examples
- Add engine to `docs/engines/tts.md`
- Check official model page for accurate feature descriptions
