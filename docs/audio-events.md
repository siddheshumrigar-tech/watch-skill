# Sound events

A transcript says what was said. It says nothing about the applause after a
demo, the fire alarm in the background, the music under an intro, or a
minute of dead air. `watch --audio-events` detects those and stores them with
timestamps, so `ask_video`, `search_videos` and `get_moment` can find them.

```bash
pip install "watch-skill[audio-events]"     # from a checkout: uv sync --extra audio-events
export WATCHSKILL_AUDIO_EVENTS_MODEL_PATH=/models/yamnet.onnx
export WATCHSKILL_AUDIO_EVENTS_LABELS_PATH=/models/yamnet_class_map.csv
watch-skill watch talk.mp4 --audio-events
```

## Labels

Every event has a `start`, an `end`, a coarse `label`, a `confidence` in
[0, 1], the classifier's own class name in `raw_label`, and the `backend`
that produced it.

| Label | Comes from |
|---|---|
| `speech` | AudioSet speech, conversation and narration classes |
| `music` | Music, singing and instrument classes |
| `applause` | Applause, clapping, cheering |
| `laughter` | Laughter and its variants |
| `alarm` | Alarms, sirens, buzzers, beeps, ringtones, smoke detectors |
| `silence` | Audio below −50 dBFS RMS. Measured, no model needed; `backend` is `rms`, `raw_label` is null |

A class that maps to none of these (traffic, typing, a dog) is not reported.
Windows are 0.96 s long every 0.48 s; adjacent windows with the same label
merge into one event and keep the highest confidence.

## The model

The shipped backend runs [YAMNet](https://tfhub.dev/google/yamnet/1) (521
AudioSet classes, about 4 MB, CPU) through onnxruntime. Watch Skill does not
download it: the provenance and hash of third-party ONNX weights cannot be
pinned here, so the operator supplies both files.

1. Export TF-Hub YAMNet to ONNX with
   [tf2onnx](https://github.com/onnx/tensorflow-onnx), for example
   `python -m tf2onnx.convert --saved-model yamnet_saved_model --output yamnet.onnx`.
   The result must have exactly one input, a rank-1 float32 waveform at
   16 kHz. Any other signature is refused with `audio_events.unsupported_model`.
2. Take `yamnet_class_map.csv` (`index,mid,display_name`) from the same
   YAMNet release.
3. Point `WATCHSKILL_AUDIO_EVENTS_MODEL_PATH` and
   `WATCHSKILL_AUDIO_EVENTS_LABELS_PATH` at them.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `WATCHSKILL_AUDIO_EVENTS_ENABLED` | `false` | Detect sound events on every watch, without passing the flag |
| `WATCHSKILL_AUDIO_EVENTS_MODEL_PATH` | unset | The YAMNet ONNX file |
| `WATCHSKILL_AUDIO_EVENTS_LABELS_PATH` | unset | Its class map |
| `WATCHSKILL_AUDIO_EVENTS_MIN_CONFIDENCE` | `0.3` | Lowest classifier score reported |

## When a piece is missing

A missing piece never fails the watch. The transcript and frames come back as
usual, silence is still detected when numpy is installed, and the summary's
`warnings` say what happened:

| Code | Meaning | Fix |
|---|---|---|
| `audio_events.missing_dependency` | onnxruntime or numpy is not installed | `pip install "watch-skill[audio-events]"` |
| `audio_events.model_not_configured` | The model or labels path is unset or missing | Set both paths (above) |
| `audio_events.unsupported_model` | The ONNX file is not a single-waveform YAMNet export | Re-export as in step 1 |
| `audio_events.decode_failed` | ffmpeg could not read the audio | Re-acquire the source; `watch-skill doctor` |

## Where they show up

- The report's `## Sounds` section and the summary's `audio_events.count` /
  `audio_events.labels`.
- REST `/v1/watch` returns the events in `audio_events`.
- The index (schema v12, table `audio_events`) and the search text
  `[sound: applause] Applause`, so `search_videos("applause")` returns hits
  of kind `audio_event`.
- `get_moment` lists overlapping events under `Sounds:`.
