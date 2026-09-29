#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from bz_ggml import serialize_breeze_voice, deserialize_breeze_voice

def test_voice_serialization_roundtrip():
    original_text = "This is a test of in-memory voice container serialization."
    frames = 125  # 10.0 seconds of audio
    codes = list(range(frames * 16))

    # Serialize to bytes in-memory
    data_bytes = serialize_breeze_voice(
        text=original_text,
        codes=codes,
        frames=frames,
        sample_rate=24000,
        n_codebooks=16
    )

    assert isinstance(data_bytes, bytes)
    assert data_bytes.startswith(b"BRZV")
    print(f"Serialized {frames} frames to {len(data_bytes)} bytes.")

    # Deserialize back from bytes
    deserialized = deserialize_breeze_voice(data_bytes)

    assert deserialized["text"] == original_text
    assert deserialized["frames"] == frames
    assert deserialized["sample_rate"] == 24000
    assert deserialized["n_codebooks"] == 16
    assert deserialized["codes"] == codes

    print("✅ In-Memory Voice Container Roundtrip Test PASSED!")

if __name__ == "__main__":
    test_voice_serialization_roundtrip()
