import unittest
import threading
import time
import math
from bz_ggml import (
    estimate_conversion_profile,
    VoiceConversionTask,
    DualInstanceCluster,
    ClusterResult
)
from bz_ggml.cluster import ConversionWorkerPool


class TestMode4DynamicBudgetingAndConcurrency(unittest.TestCase):

    def test_estimate_conversion_profile_exact_math(self):
        """
        Verify exact deterministic token budgeting formula:
          total_c = 1 + ref_text_tokens + (ref_frames + 1) + 1 + text_tokens
          st_c_capacity = total_c + src_frames + 8
          st_u_capacity = total_u + src_frames + 8 (if cfg_scale > 1.0)
        """
        # Test Case 1: 8.32 seconds audio = 104 frames (12.5 fps)
        # Reference voice = 4 seconds = 50 frames
        # CFG scale = 1.5 (dual branch)
        prof = estimate_conversion_profile(
            src_frames=104,
            ref_frames=50,
            ref_text="Hello world reference voice prompt",
            src_text=None, # auto filler text
            cfg_scale=1.5
        )

        # 5 words in ref_text -> ceil(5 * 1.30) = 7 tokens
        self.assertEqual(prof["src_frames"], 104)
        self.assertEqual(prof["ref_frames"], 50)
        
        # Filler text for 104 frames (8.32s): ceil(8.32 * 4.0) + 4 = 34 + 4 = 38
        expected_text_tokens = int(math.ceil((104 / 12.5) * 4.0)) + 4
        self.assertEqual(prof["text_tokens"], expected_text_tokens)

        # total_c = [S0] (1) + ref_text (7) + ref_codes (50 + 1 eos) + [S0] (1) + text (38) = 98
        total_c = 1 + 7 + 51 + 1 + expected_text_tokens
        expected_st_c = total_c + 104 + 8
        self.assertEqual(prof["st_c_capacity"], expected_st_c)

        # total_u = [S0] (1) + text (38) = 39
        total_u = 1 + expected_text_tokens
        expected_st_u = total_u + 104 + 8
        self.assertEqual(prof["st_u_capacity"], expected_st_u)

        # With CFG: sum of both branches
        self.assertEqual(prof["total_needed"], expected_st_c + expected_st_u)
        
        # Verify VRAM is tiny (~35 MiB)
        self.assertLess(prof["vram_estimate_mib"], 100.0)
        self.assertGreater(prof["vram_estimate_mib"], 10.0)

    def test_estimate_conversion_profile_no_cfg(self):
        """
        Verify that with cfg_scale=1.0 (CFG off), unconditional branch is 0 tokens.
        """
        prof = estimate_conversion_profile(
            src_frames=100,
            ref_frames=40,
            ref_text="Test prompt",
            src_text="Spoken words here",
            cfg_scale=1.0
        )
        self.assertEqual(prof["st_u_capacity"], 0)
        self.assertEqual(prof["total_needed"], prof["st_c_capacity"])

    def test_conversion_worker_pool_concurrency_ceiling(self):
        """
        Verify strict concurrency limiter (max 2 concurrent requests).
        """
        pool = ConversionWorkerPool(max_concurrent=2)
        self.assertEqual(pool.active_count, 0)

        # Acquire 1st slot
        self.assertTrue(pool.acquire(blocking=False))
        self.assertEqual(pool.active_count, 1)

        # Acquire 2nd slot
        self.assertTrue(pool.acquire(blocking=False))
        self.assertEqual(pool.active_count, 2)

        # 3rd attempt must fail immediately when non-blocking
        self.assertFalse(pool.acquire(blocking=False))
        self.assertEqual(pool.active_count, 2)

        # Release one slot
        pool.release()
        self.assertEqual(pool.active_count, 1)

        # 3rd attempt now succeeds
        self.assertTrue(pool.acquire(blocking=False))
        self.assertEqual(pool.active_count, 2)

        # Clean up
        pool.release()
        pool.release()
        self.assertEqual(pool.active_count, 0)

    def test_multi_threaded_concurrency_throttling(self):
        """
        Launch 6 concurrent threads trying to perform conversion.
        Verify peak simultaneous active workers NEVER exceeds max_concurrent (2).
        """
        pool = ConversionWorkerPool(max_concurrent=2)
        peak_active = 0
        peak_lock = threading.Lock()
        completed_count = 0

        def worker_task():
            nonlocal peak_active, completed_count
            pool.acquire(blocking=True)
            try:
                with peak_lock:
                    if pool.active_count > peak_active:
                        peak_active = pool.active_count
                time.sleep(0.05) # Simulate audio processing duration
            finally:
                pool.release()
                with peak_lock:
                    completed_count += 1

        threads = [threading.Thread(target=worker_task) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(completed_count, 6)
        self.assertLessEqual(peak_active, 2)
        self.assertEqual(pool.active_count, 0)

    def test_voice_conversion_task_creation(self):
        """
        Verify VoiceConversionTask initialization and defaults.
        """
        dummy_codes = [0] * (16 * 50)
        task = VoiceConversionTask(
            id=101,
            src_codes=dummy_codes,
            src_frames=50,
            ref_text="My voice",
            cfg_scale=1.5,
            keep_acoustic=2,
            feed_source=True
        )
        self.assertEqual(task.id, 101)
        self.assertEqual(task.src_frames, 50)
        self.assertEqual(task.keep_acoustic, 2)
        self.assertTrue(task.feed_source)
        self.assertFalse(task.client_has_vocoder)


if __name__ == "__main__":
    unittest.main()
