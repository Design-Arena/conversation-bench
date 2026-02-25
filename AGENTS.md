## Recent Learnings

- **Realtime timeout false empties** -> Drive both watchdog extension and `_check_no_response()` from the same raw downstream activity signal (`TTSAudioRawFrame`, `TTSTextFrame`, `LLMTextFrame`, and bot speaking frames), not just coarse TTS/bot state -> Prevents `[EMPTY_RESPONSE]` retries when output is already in flight but state bookkeeping lags under load.
- **Rehydrated review context drift** -> Show prior turns from benchmark gold history, including `required_function_call` and `function_call_response`, rather than prior model outputs -> Keeps review context aligned with single-step rehydration semantics and avoids misleading failure analysis.
