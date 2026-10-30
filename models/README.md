# models/

| Path | What it is |
|---|---|
| `basic_pitch/nmp.onnx` | Spotify Basic Pitch (ICASSP 2022) note-transcription model, ONNX export, 230 KB, Apache-2.0. Used by the polyphonic and full-mix modes through `onnxruntime`. See `basic_pitch/NOTICE.md`. |

Nothing else needs to live here:

* **pYIN** (monophonic mode) is an algorithm in `librosa`; it has no weights.
* **Demucs** (optional source separation) downloads its weights (~80 MB) to the PyTorch hub cache on
  first use, not to this folder.

Custom or fine-tuned models can be dropped here later; everything except the files listed above is
ignored by git.
