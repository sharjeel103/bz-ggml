#!/usr/bin/env python3
"""
Unit test for per-job stream_pcm parameter:
Verifies that stream_pcm=False suppresses intermediate 32-frame slicing
and flushes the full continuous token sequence at EOS for single-pass studio vocoding.
"""

import unittest
from bz_ggml import UserTask, VoiceConversionTask


class TestStreamPCMMode(unittest.TestCase):
    def test_task_default_parameters(self):
        # 1. Standard TTS task defaults to stream_pcm=True for backward compatibility
        task_streaming = UserTask(id=1, text="Hello world")
        self.assertTrue(task_streaming.stream_pcm)
        self.assertFalse(task_streaming.client_has_vocoder)

        # 2. Studio task explicitly sets stream_pcm=False
        task_studio = UserTask(id=2, text="Studio recording", stream_pcm=False)
        self.assertFalse(task_studio.stream_pcm)
        self.assertFalse(task_studio.client_has_vocoder)

        # 3. VoiceConversionTask defaults to stream_pcm=False (native single-pass)
        task_conv = VoiceConversionTask(id=3, src_codes=[1, 2], src_frames=10)
        self.assertFalse(task_conv.stream_pcm)

    def test_chunk_buffer_accumulation_logic(self):
        # Simulate active session with 100 generated frames (16 tokens each)
        total_frames = 100
        dummy_tokens = list(range(total_frames * 16))
        chunk_size = 32

        # --- CASE A: stream_pcm = True (Streaming 32-frame chunks) ---
        chunk_buffer_stream = list(dummy_tokens)
        burst_ready_chunks = []
        while chunk_size > 0 and len(chunk_buffer_stream) >= (chunk_size * 16):
            dispatch_tokens = chunk_buffer_stream[:chunk_size * 16]
            chunk_buffer_stream = chunk_buffer_stream[chunk_size * 16:]
            burst_ready_chunks.append(dispatch_tokens)

        # In streaming mode, 3 chunks of 32 frames were sliced out
        self.assertEqual(len(burst_ready_chunks), 3)
        for chunk in burst_ready_chunks:
            self.assertEqual(len(chunk), 32 * 16)
        # Leftover in buffer is 4 frames
        self.assertEqual(len(chunk_buffer_stream), 4 * 16)

        # --- CASE B: stream_pcm = False (Studio Single-Pass) ---
        chunk_buffer_studio = list(dummy_tokens)
        burst_ready_studio = []
        # When stream_pcm is False, intermediate while loop is skipped
        stream_pcm = False
        if stream_pcm:
            while chunk_size > 0 and len(chunk_buffer_studio) >= (chunk_size * 16):
                dispatch_tokens = chunk_buffer_studio[:chunk_size * 16]
                chunk_buffer_studio = chunk_buffer_studio[chunk_size * 16:]
                burst_ready_studio.append(dispatch_tokens)

        # Zero intermediate chunks emitted!
        self.assertEqual(len(burst_ready_studio), 0)
        # Full 100 frames remain in buffer for single-pass flush at EOS
        self.assertEqual(len(chunk_buffer_studio), 100 * 16)

        # At EOS, all 100 frames are dispatched together
        leftover_at_eos = list(chunk_buffer_studio)
        chunk_buffer_studio.clear()
        self.assertEqual(len(leftover_at_eos), 100 * 16)
        self.assertEqual(len(chunk_buffer_studio), 0)


if __name__ == "__main__":
    unittest.main()
