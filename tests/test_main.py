import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import main as analyzer


class VideoProcessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.video = Path(self.directory.name) / 'fixture.avi'
        self.image = Path(self.directory.name) / 'court.png'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 1000, (64, 48))
        self.assertTrue(writer.isOpened())
        try:
            for intensity in (20, 80):
                writer.write(np.full((48, 64, 3), intensity, dtype=np.uint8))
        finally:
            writer.release()
        self.assertTrue(cv2.imwrite(str(self.image), np.full((65, 40, 3), 60, dtype=np.uint8)))

    def run_video(self, image=None, display_error=None, fps=None):
        capture = cv2.VideoCapture(str(self.video))
        self.addCleanup(capture.release)
        wrapped_capture = mock.Mock(wraps=capture)
        wrapped_capture.set.return_value = False
        if fps is not None:
            wrapped_capture.get.return_value = fps
        output = io.StringIO()
        with mock.patch.object(analyzer, 'VIDEO_DOSYA_YOL', str(self.video)), \
                mock.patch.object(analyzer, 'KORT_RESIM_YOL', str(image or self.image)), \
                mock.patch.object(cv2, 'VideoCapture', return_value=wrapped_capture), \
                mock.patch.object(cv2, 'imshow', side_effect=display_error) as display, \
                mock.patch.object(cv2, 'waitKey', return_value=ord('+')) as wait, \
                mock.patch.object(cv2, 'destroyAllWindows') as close, \
                contextlib.redirect_stdout(output):
            if display_error is None:
                analyzer.main()
            else:
                with self.assertRaisesRegex(RuntimeError, 'display failed'):
                    analyzer.main()
        wrapped_capture.release.assert_called_once()
        close.assert_called_once()
        self.assertFalse(capture.isOpened())
        return display, wait, output.getvalue()

    def test_bundled_asset_paths_work_outside_repository(self):
        previous_directory = os.getcwd()
        try:
            os.chdir(self.directory.name)
            self.assertTrue(Path(analyzer.VIDEO_DOSYA_YOL).is_file())
            self.assertIsNotNone(cv2.imread(analyzer.KORT_RESIM_YOL))
        finally:
            os.chdir(previous_directory)

    def test_all_frames_render_when_seeking_is_unsupported(self):
        display, wait, _ = self.run_video()
        frames = [call.args[1] for call in display.call_args_list if call.args[0] == 'Tenis Mac Analizi']
        maps = [call.args[1] for call in display.call_args_list if call.args[0] == 'Kort']
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(maps), 2)
        self.assertEqual(frames[0].shape, (600, 800, 3))
        self.assertEqual(maps[0].shape, (650, 400, 3))
        self.assertLess(float(frames[0][300, 400].mean()), float(frames[1][300, 400].mean()))
        self.assertTrue(all(call.args[0] >= 1 for call in wait.call_args_list))

    def test_invalid_fps_uses_fallback(self):
        for fps in (0, float('nan'), float('inf')):
            with self.subTest(fps=fps):
                _, wait, _ = self.run_video(fps=fps)
                self.assertEqual(wait.call_args_list[0].args[0], 33)

    def test_corrupt_image_falls_back_without_aborting_video(self):
        self.image.write_bytes(b'not an image')
        display, wait, output = self.run_video()
        self.assertIn('court image could not be decoded', output)
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(display.call_count, 4)

    def test_missing_image_falls_back_without_aborting_video(self):
        _, wait, output = self.run_video(image=Path(self.directory.name) / 'absent.png')
        self.assertIn('court image not found', output)
        self.assertEqual(wait.call_count, 2)

    def test_capture_and_windows_close_on_processing_exception(self):
        self.run_video(display_error=RuntimeError('display failed'))

    def test_corrupt_video_releases_capture(self):
        self.video.write_bytes(b'not a video')
        _, wait, output = self.run_video()
        self.assertIn('video file could not be decoded', output)
        wait.assert_not_called()

    def test_missing_video_does_not_open_capture(self):
        with mock.patch.object(analyzer, 'VIDEO_DOSYA_YOL', str(self.video) + '.missing'), \
                mock.patch.object(cv2, 'VideoCapture') as capture, \
                contextlib.redirect_stdout(io.StringIO()):
            analyzer.main()
        capture.assert_not_called()


class MiniMapTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((650, 400, 3), 60, dtype=np.uint8)
        self.homography = np.eye(3, dtype=np.float64)

    def tracker(self, box, side):
        tracker = analyzer.TenisAnalizi(box)
        tracker.kort_tarafi = side
        return tracker

    def render(self, trackers):
        return analyzer.kort_krokisini_ciz(
            self.background, self.homography, trackers, [], 1, analyzer.GORSEL_PARAMETRELER
        )

    def test_remaining_player_keeps_own_position_when_other_disappears(self):
        upper = self.tracker((40, 70, 20, 30), 'ust')
        lower = self.tracker((290, 470, 20, 30), 'alt')
        self.render({'ust': upper, 'alt': lower})
        rendered = self.render({'ust': None, 'alt': lower})
        self.assertEqual(tuple(rendered[500, 300]), (255, 255, 255))
        self.assertEqual(tuple(rendered[380, 225]), (60, 60, 60))

    def test_replacement_tracker_does_not_inherit_previous_position(self):
        self.render({'alt': self.tracker((290, 470, 20, 30), 'alt')})
        replacement = self.tracker((40, 570, 20, 30), 'alt')
        rendered = self.render({'alt': replacement})
        self.assertEqual(tuple(rendered[600, 50]), (255, 255, 255))

    def test_first_player_position_is_clamped_to_visible_map(self):
        lower = self.tracker((500, 800, 20, 30), 'alt')
        rendered = self.render({'alt': lower})
        self.assertEqual(tuple(rendered[620, 390]), (255, 255, 255))
        np.testing.assert_array_equal(self.background, np.full((650, 400, 3), 60, dtype=np.uint8))


class DetectionFixtureTests(unittest.TestCase):
    def test_legacy_hough_line_layout_remains_supported(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        corners = np.array([[120, 50], [280, 50], [360, 250], [40, 250]], dtype=np.int32)
        cv2.polylines(frame, [corners], True, (255, 255, 255), 3)
        hough = cv2.HoughLinesP

        def legacy_layout(*args, **kwargs):
            return hough(*args, **kwargs).reshape(-1, 1, 4)

        with mock.patch.object(cv2, 'HoughLinesP', side_effect=legacy_layout):
            detected = analyzer.roi_bolgesinde_kort_tespit_et(frame)
        np.testing.assert_allclose(detected, corners, atol=10)

    def test_synthetic_court_lines_produce_valid_geometry(self):
        frame = np.zeros((300, 400, 3), dtype=np.uint8)
        corners = np.array([[120, 50], [280, 50], [360, 250], [40, 250]], dtype=np.int32)
        cv2.polylines(frame, [corners], True, (255, 255, 255), 3)
        detected = analyzer.roi_bolgesinde_kort_tespit_et(frame)
        self.assertIsNotNone(detected)
        np.testing.assert_allclose(detected, corners, atol=10)
        self.assertTrue(analyzer.kort_geometrisini_dogrula(
            detected, analyzer.UST_ALT_ORAN_ARALIK, analyzer.EN_BOY_ORAN_ARALIK,
            analyzer.DIKEY_CIZGI_ACI_ARALIK, analyzer.YAN_YUKSEKLIK_ORAN_ARALIK
        ))

    def test_ball_detection_rejects_candidate_inside_player(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (98, 98), (102, 102), 255, -1)
        court = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype=np.int32)
        detected = analyzer.top_tespit_et(mask, None, 0, court, [], analyzer.TOP_PARAMETRELERI)
        self.assertEqual(detected[1], (100, 100))
        player = analyzer.TenisAnalizi((90, 80, 20, 40))
        self.assertIsNone(analyzer.top_tespit_et(mask, None, 0, court, [player], analyzer.TOP_PARAMETRELERI))


if __name__ == '__main__':
    unittest.main()
