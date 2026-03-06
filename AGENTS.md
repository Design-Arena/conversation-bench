## Recent Learnings

- **Rehydrated turn-taking analysis is misleading -> Re-judge `--rehydrate` runs with `--skip-turn-taking` and preserve any raw turn-taking summaries separately -> Transcript rows accumulate across all rehydrated turns, but the realtime pipeline writes a single `conversation.wav` in the run root, so the audio analyzer does not see a true full-conversation waveform.**
