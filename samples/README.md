# samples/

Two short synthetic recordings for a first test run (no copyright, generated with the same
additive synthesiser the test-suite uses):

* `demo_melody_monophonic.wav` - a single-voice tune with repeated notes (C4 D4 E4 C4 C4 ...).
  Analyse it with **Mode A - Monophonic**; every note should come out as its own event.
* `demo_chords_polyphonic.wav` - a I-IV-V-I progression with a bass note and overlapping
  sustain. Analyse it with **Mode B - Polyphonic**; the piano roll should show 4 simultaneous
  notes per chord, with a few low-confidence harmonic "ghost" notes you can hide with the
  confidence slider.

Use *Upload & Analyze -> "Or analyse files that are already on this computer"* and paste the
path of this folder to load both at once.
