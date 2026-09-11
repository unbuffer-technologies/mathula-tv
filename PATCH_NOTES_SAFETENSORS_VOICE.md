# Native voice classifier safetensors dependency fix

- Adds `safetensors>=0.8,<1` to the `speaker-recognition` optional dependencies.
- Preflights `safetensors` before loading the JaesungHuh ECAPA voice-family model.
- Prevents Hugging Face `PyTorchModelHubMixin` from surfacing the misleading `name 'safetensors' is not defined` error.
- Installer now fails immediately with the exact dependency command when safetensors is absent.
