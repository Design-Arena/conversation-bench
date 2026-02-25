## Recent Learnings

- **Realtime watchdog false empties** -> Track raw downstream output activity (`TTSAudioRawFrame`, `TTSTextFrame`, `LLMTextFrame`, and bot speaking frames) in `TurnGate`, not just coarse TTS/bot state -> Prevents `[EMPTY_RESPONSE]` retries when the model is already producing output but state bookkeeping lags under load.
