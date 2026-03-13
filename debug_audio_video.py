import json
import os
import traceback

import api_server


def main() -> None:
    print("cwd", os.getcwd())

    # Ensure app.py helpers are loaded
    api_server._try_load_app()  # type: ignore[attr-defined]

    print(
        "has_helpers",
        bool(getattr(api_server, "_predict_from_audio_path", None)),
        bool(getattr(api_server, "_load_audio_models", None)),
    )

    # Load audio models if available
    load_audio = getattr(api_server, "_load_audio_models", None)
    audio_models = None
    if callable(load_audio):
        try:
            audio_models = load_audio()
            print("audio_models_keys", list(audio_models.keys()) if audio_models else None)
        except Exception as e:  # noqa: BLE001
            print("Error loading audio models:", repr(e))

    audio_func = getattr(api_server, "_predict_from_audio_path", None)
    if callable(audio_func):
        print("\n=== Testing predict_from_audio_path('test_audio.wav') ===")
        try:
            res = audio_func("test_audio.wav", audio_models=audio_models)
            print("AUDIO RESULT", json.dumps(res, indent=2))
        except Exception as e:  # noqa: BLE001
            print("AUDIO ERROR", repr(e))
            traceback.print_exc()
    else:
        print("No _predict_from_audio_path function found on api_server")


if __name__ == "__main__":
    main()

