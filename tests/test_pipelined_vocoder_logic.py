"""
Unit test for pipelined chunk-wise vocoder logic and context passing.
Verifies:
  1. Contiguous slice scheduling from out-of-order chunk completions.
  2. Trailing 80-frame acoustic context passing without bleeding or gaps.
  3. Ordered multi-part PCM concatenation and bit-exact sample counts.
  4. Token capacity coordinator recalibration (26,500 / 35,000 / 18,500).
"""

import sys, os, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from bz_ggml.cluster import TokenCapacityCoordinator
from bz_ggml.orchestrator import RequestTTSAssembly, VocoderChunkJob


class TestPipelinedVocoderLogic(unittest.TestCase):

    def test_token_capacity_ceilings(self):
        # Pure TTS (ASR disabled): GPU 0 and GPU 1 both get full 33,100 token pools
        coord_tts = TokenCapacityCoordinator(precision="int8", enable_asr=False, enable_enhancer=True)
        self.assertEqual(coord_tts.get_admission_ceiling(0), 33100)
        self.assertEqual(coord_tts.get_admission_ceiling(1), 33100)
        self.assertEqual(coord_tts.ceiling_gpu0, 33100)
        self.assertEqual(coord_tts.ceiling_gpu1_normal, 33100)

        # Full Cluster with ASR (GPU 0): GPU 0 reserves 2,160 MiB for Whisper -> 24,400 tokens
        coord_asr = TokenCapacityCoordinator(precision="int8", enable_asr=True, enable_enhancer=True)
        self.assertEqual(coord_asr.get_admission_ceiling(0), 24400)
        self.assertEqual(coord_asr.get_admission_ceiling(1), 33100)
        self.assertEqual(coord_asr.ceiling_gpu0, 24400)
        self.assertEqual(coord_asr.ceiling_gpu1_normal, 33100)
        self.assertEqual(coord_asr.ceiling_gpu1_enh, 12900)

        # Test Enhancer headroom clamp
        coord_asr.enhancer_active = True
        self.assertEqual(coord_asr.get_admission_ceiling(1), 12900)
        coord_asr.enhancer_active = False
        self.assertEqual(coord_asr.get_admission_ceiling(1), 33100)

    def test_assembly_in_order_chunk_scheduling(self):
        chunks_text = ["chunk0", "chunk1", "chunk2"]
        assembly = RequestTTSAssembly(request_id="test_req", chunks_text=chunks_text)

        # Chunk 0 arrives (100 frames = 1,600 ints)
        tokens_0 = list(range(100 * 16))
        jobs_0 = assembly.record_chunk_tokens(0, tokens_0)
        self.assertEqual(len(jobs_0), 1)
        self.assertEqual(jobs_0[0].job_seq, 0)
        self.assertEqual(len(jobs_0[0].new_tokens), 100 * 16)
        self.assertEqual(len(jobs_0[0].ctx_tokens), 0)  # First slice has 0 context
        self.assertFalse(jobs_0[0].is_final)

        # Trailing context should be last 80 frames of tokens_0
        expected_ctx = tokens_0[-80 * 16 :]
        self.assertEqual(assembly.last_ctx_tokens, expected_ctx)

        # Chunk 1 arrives (200 frames = 3,200 ints)
        tokens_1 = [1000 + x for x in range(200 * 16)]
        jobs_1 = assembly.record_chunk_tokens(1, tokens_1)
        self.assertEqual(len(jobs_1), 1)
        self.assertEqual(jobs_1[0].job_seq, 1)
        self.assertEqual(jobs_1[0].ctx_tokens, expected_ctx)
        self.assertEqual(len(jobs_1[0].new_tokens), 200 * 16)
        self.assertFalse(jobs_1[0].is_final)

        # Chunk 2 arrives (150 frames = 2,400 ints) -> Final chunk!
        expected_ctx_2 = tokens_1[-80 * 16 :]
        tokens_2 = [2000 + x for x in range(150 * 16)]
        jobs_2 = assembly.record_chunk_tokens(2, tokens_2)
        self.assertEqual(len(jobs_2), 1)
        self.assertEqual(jobs_2[0].job_seq, 2)
        self.assertEqual(jobs_2[0].ctx_tokens, expected_ctx_2)
        self.assertEqual(len(jobs_2[0].new_tokens), 150 * 16)
        self.assertTrue(jobs_2[0].is_final)
        self.assertTrue(assembly.all_chunks_queued)

        # Simulate vocoder completion out-of-order (Part 1, then Part 0, then Part 2)
        pcm_0 = np.full(100 * 1920, 0.1, dtype=np.float32)
        pcm_1 = np.full(200 * 1920, 0.2, dtype=np.float32)
        pcm_2 = np.full(150 * 1920, 0.3, dtype=np.float32)

        is_done, _, _ = assembly.record_pcm_part(1, pcm_1)
        self.assertFalse(is_done)

        is_done, _, _ = assembly.record_pcm_part(0, pcm_0)
        self.assertFalse(is_done)

        is_done, full_pcm, merged_toks = assembly.record_pcm_part(2, pcm_2)
        self.assertTrue(is_done)
        self.assertIsNotNone(full_pcm)
        self.assertEqual(len(full_pcm), (100 + 200 + 150) * 1920)

        # Verify ordered concatenation
        self.assertTrue(np.all(full_pcm[: 100 * 1920] == 0.1))
        self.assertTrue(np.all(full_pcm[100 * 1920 : 300 * 1920] == 0.2))
        self.assertTrue(np.all(full_pcm[300 * 1920 :] == 0.3))
        self.assertEqual(len(merged_toks), (100 + 200 + 150) * 16)

    def test_assembly_out_of_order_chunk_scheduling(self):
        """Tests that out-of-order chunk arrivals strictly gate vocoder jobs until contiguous."""
        chunks_text = ["c0", "c1", "c2", "c3"]
        assembly = RequestTTSAssembly(request_id="test_ooo", chunks_text=chunks_text)

        tokens = [list(range(i * 1000, i * 1000 + 100 * 16)) for i in range(4)]

        # Chunk 0 arrives
        jobs = assembly.record_chunk_tokens(0, tokens[0])
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].job_seq, 0)
        self.assertFalse(jobs[0].is_final)

        # Chunk 2 arrives (Chunk 1 still pending!)
        jobs = assembly.record_chunk_tokens(2, tokens[2])
        self.assertEqual(len(jobs), 0)  # Cannot queue chunk 2 without chunk 1!

        # Chunk 3 arrives (Chunk 1 still pending!)
        jobs = assembly.record_chunk_tokens(3, tokens[3])
        self.assertEqual(len(jobs), 0)  # Cannot queue chunk 3 without chunk 1!

        # Chunk 1 finishes! Now contiguous slice [1, 2, 3] should immediately yield 3 jobs!
        jobs = assembly.record_chunk_tokens(1, tokens[1])
        self.assertEqual(len(jobs), 3)

        self.assertEqual(jobs[0].job_seq, 1)
        self.assertEqual(jobs[0].new_tokens, tokens[1])
        self.assertEqual(jobs[0].ctx_tokens, tokens[0][-80 * 16 :])
        self.assertFalse(jobs[0].is_final)

        self.assertEqual(jobs[1].job_seq, 2)
        self.assertEqual(jobs[1].new_tokens, tokens[2])
        self.assertEqual(jobs[1].ctx_tokens, tokens[1][-80 * 16 :])
        self.assertFalse(jobs[1].is_final)

        self.assertEqual(jobs[2].job_seq, 3)
        self.assertEqual(jobs[2].new_tokens, tokens[3])
        self.assertEqual(jobs[2].ctx_tokens, tokens[2][-80 * 16 :])
        self.assertTrue(jobs[2].is_final)
        self.assertTrue(assembly.all_chunks_queued)


if __name__ == "__main__":
    unittest.main()
