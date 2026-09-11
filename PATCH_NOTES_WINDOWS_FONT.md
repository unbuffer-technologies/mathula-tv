# Native publication Windows font fix

- Removes the hard-coded Linux-only DejaVuSans-Bold.ttf requirement from `tiktok_editor.py`.
- Resolution order: `MATHULA_TV_BOLD_FONT`, Windows `%WINDIR%\\Fonts` Segoe UI/Arial/Calibri, Linux DejaVu/Liberation/FreeSans, then Pillow default.
- The renderer no longer aborts merely because a preferred font file is absent.
- No font files are bundled.
- Existing native editorial response checkpoints remain reusable; rerunning native-dub does not require a fresh editorial Grok call unless forced or the checkpoint is invalid.
