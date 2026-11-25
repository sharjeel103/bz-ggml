import unittest
from bz_ggml import (
    UserTask,
    VoiceConversionTask,
    split_text_into_smart_chunks,
    split_text_with_anchor,
    format_chunk_preview_markdown,
    EngineOrchestrator,
)
from bz_ggml.bindings import BreezeLib


class TestProductionFixes(unittest.TestCase):

    def test_user_task_new_attributes(self):
        task = UserTask(
            id=1,
            text="Hello world",
            depth_temperature=0.0,
            depth_cfg_codebooks=15
        )
        self.assertEqual(task.depth_temperature, 0.0)
        self.assertEqual(task.depth_cfg_codebooks, 15)
        self.assertFalse(task.stream_pcm)

    def test_split_text_with_anchor(self):
        # Short text should remain 1 chunk
        short_text = "This is a short sentence."
        chunks_short = split_text_with_anchor(short_text, anchor_target_words=35, anchor_max_words=50)
        self.assertEqual(len(chunks_short), 1)
        self.assertEqual(chunks_short[0], short_text)

        # Long text (> 50 words) should yield anchor + follower chunks
        long_text = "Sentence number one with some words. " * 20  # ~140 words
        chunks_long = split_text_with_anchor(long_text, anchor_target_words=35, anchor_max_words=50, target_words=60, max_words=80)
        self.assertGreater(len(chunks_long), 1)
        anchor_wc = len(chunks_long[0].split())
        self.assertLessEqual(anchor_wc, 50)

    def test_orchestrator_prepare_tts_chunks_guard(self):
        orch = EngineOrchestrator.__new__(EngineOrchestrator)
        orch.SINGLE_CHUNK_MAX_WORDS = 200
        orch.SINGLE_CHUNK_MAX_TOKENS = 2000

        # <= 200 words fits in single chunk
        short_script = "Natural sounding speech. " * 30  # ~90 words
        chunks_short = orch._prepare_tts_chunks(short_script)
        self.assertEqual(len(chunks_short), 1)

        # > 200 words triggers anchor splitting
        long_script = "Natural sounding speech with high quality cadence. " * 35  # ~245 words
        chunks_long = orch._prepare_tts_chunks(long_script)
        self.assertGreater(len(chunks_long), 1)

    def test_breeze_lib_c_signatures_loaded(self):
        # Verify BreezeLib ctypes signature definitions
        lib = BreezeLib.__new__(BreezeLib)
        # Mock handle for _bind_functions
        class DummyFunc:
            argtypes = None
            restype = None
        class DummyCDLL:
            def __getattr__(self, name):
                f = DummyFunc()
                setattr(self, name, f)
                return f
        lib.lib = DummyCDLL()
        lib._bind_functions()
        # Ensure new function signatures are configured
        self.assertIsNotNone(lib.lib.breeze_generator_session_create_v2.argtypes)
        self.assertEqual(len(lib.lib.breeze_generator_session_create_v2.argtypes), 18)
        self.assertIsNotNone(lib.lib.breeze_vocoder_decode_tiled.argtypes)
        self.assertEqual(len(lib.lib.breeze_vocoder_decode_tiled.argtypes), 6)


if __name__ == "__main__":
    unittest.main()
