# Basic Pitch model - provenance and license

`nmp.onnx` is the ONNX export of Spotify's **Basic Pitch** note-transcription model
(ICASSP 2022, "nmp"), taken unchanged from the `basic-pitch` 0.4.0 Python package
(`basic_pitch/saved_models/icassp_2022/nmp.onnx`).

* Source: https://github.com/spotify/basic-pitch
* Copyright 2022 Spotify AB
* License: Apache License 2.0 (see `LICENSE` in this folder)
* SHA-256: `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`
* Paper: Bittner, Bosch, Rubinstein, Velichko, Ewert, "A Lightweight Instrument-Agnostic Model for
  Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022.

NoteVision AI runs the model with `onnxruntime` (`src/basic_pitch_onnx.py`), which also contains a
port of the package's note-decoding algorithm under the same Apache 2.0 terms. No modifications were
made to the model weights.
