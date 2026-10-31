import io
import unittest
from unittest.mock import Mock, patch

from time_lapser import FRAME_SIZE, build_segments, find_activity, render_video


class BuildSegmentsTests(unittest.TestCase):
    def test_no_detected_activity_keeps_entire_video(self):
        self.assertEqual(build_segments([], 30.0, 8.0, 1.5), [(0.0, 30.0)])

    def test_joins_activity_with_short_pauses_and_keeps_padding(self):
        self.assertEqual(
            build_segments([2.0, 5.0, 19.0], 24.0, 8.0, 1.5),
            [(0.0, 6.5), (16.5, 20.5)],
        )

    def test_clips_padding_to_video_boundaries(self):
        self.assertEqual(
            build_segments([0.5, 9.5], 10.0, 10.0, 2.0),
            [(0.0, 10.0)],
        )

    def test_zero_padding_still_retains_the_sampled_change_interval(self):
        self.assertEqual(
            build_segments([5.0], 10.0, 8.0, 0.0, sample_interval=2.0),
            [(3.0, 5.0)],
        )

    def test_merges_overlapping_ranges(self):
        self.assertEqual(
            build_segments([1.0, 12.0], 20.0, 8.0, 6.0),
            [(0.0, 18.0)],
        )


class VideoProcessingTests(unittest.TestCase):
    @patch("time_lapser.subprocess.Popen")
    def test_find_activity_uses_frame_difference_and_sample_timestamps(self, popen):
        blank = bytes(FRAME_SIZE)
        changed = bytes([10]) * FRAME_SIZE
        process = Mock()
        process.stdout = io.BytesIO(blank + changed + changed)
        process.wait.return_value = 0
        popen.return_value = process

        self.assertEqual(find_activity("build.mp4", 2.0, 5.0), [2.0])
        command = popen.call_args.args[0]
        self.assertIn("fps=0.5", command[command.index("-vf") + 1])

    @patch("time_lapser.subprocess.Popen")
    def test_zero_threshold_does_not_count_identical_frames(self, popen):
        process = Mock()
        process.stdout = io.BytesIO(bytes(FRAME_SIZE * 2))
        process.wait.return_value = 0
        popen.return_value = process

        self.assertEqual(find_activity("build.mp4", 1.0, 0.0), [])

    @patch("time_lapser.subprocess.run")
    def test_render_video_concatenates_trimmed_video_and_audio(self, run):
        render_video("build.mp4", "trimmed.mp4", [(1.0, 3.0), (8.0, 10.0)], True)

        command = run.call_args.args[0]
        filter_graph = command[command.index("-filter_complex") + 1]
        self.assertIn("concat=n=2:v=1:a=1[outv][outa]", filter_graph)
        self.assertIn("[0:a:0]atrim=start=8.000000:end=10.000000", filter_graph)
        self.assertEqual(command[command.index("-map") + 1], "[outv]")


if __name__ == "__main__":
    unittest.main()
