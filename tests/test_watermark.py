import os
import sys
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import cv2

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.logic.algorithms import (
    process_lsb, process_fft, process_dct, process_dwt, _deblock_8x8
)
import src.logic.processor as processor_mod
from src.logic.processor import (
    WatermarkProcessor, blend_alpha_to_bgr, imwrite_unicode, imread_unicode, load_image_checked
)
from src.ui.main_view import get_resource_path, get_detection_level


def _make_dummy_image(h, w, c=3):
    rng = random.Random(42)
    data = bytes([rng.randint(40, 210) for _ in range(h * w * c)])
    if c == 1:
        return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w, c))


class TestWatermarkSuite(unittest.TestCase):
    def test_resource_path(self):
        p = get_resource_path("logo.png")
        self.assertTrue(os.path.exists(p), f"Resource logo.png not found: {p}")

    def test_alpha_blending_8bit_and_16bit(self):
        rgba8 = np.zeros((10, 10, 4), dtype=np.uint8)
        blended8 = blend_alpha_to_bgr(rgba8, bg_color=(255, 255, 255))
        self.assertEqual(blended8.dtype, np.uint8)
        self.assertTrue(np.all(blended8 == 255))

        rgba16 = np.zeros((10, 10, 4), dtype=np.uint16)
        blended16 = blend_alpha_to_bgr(rgba16, bg_color=(255, 255, 255))
        self.assertEqual(blended16.dtype, np.uint16)
        self.assertTrue(np.all(blended16 == 65535))

    def test_pipeline_lsb_zero_cleanliness(self):
        img = _make_dummy_image(128, 128, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            test_in = os.path.join(tmpdir, "input.png")
            test_out = os.path.join(tmpdir, "cleaned.png")
            self.assertTrue(imwrite_unicode(test_in, img))

            processor = WatermarkProcessor()
            res = processor.process_image(
                image_path=test_in,
                methods=["LSB", "FFT", "DCT", "DWT"],
                action="clean",
                color_preserve=True,
                aggressiveness="Balanced",
                output_path=test_out
            )
            msg, ok = res
            self.assertTrue(ok, f"Processing failed: {msg}")
            self.assertTrue(res.success)
            self.assertEqual(res.output_path, test_out)

            cleaned = imread_unicode(test_out)
            self.assertIsNotNone(cleaned)
            for c in range(3):
                lsb = cleaned[:, :, c] & 1
                self.assertTrue(np.all(lsb == 0), f"Channel {c} LSB was corrupted!")

    def test_fft_notch_depth(self):
        img = np.ones((64, 64), dtype=np.uint8) * 128
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        fshift[20, 20] += 1e6
        fshift[64 - 20, 64 - 20] += 1e6
        img_spiked = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(fshift))), 0, 255).astype(np.uint8)

        cleaned = process_fft(img_spiked, "Balanced", "clean")
        f_clean = np.fft.fftshift(np.fft.fft2(cleaned))
        self.assertLess(np.abs(f_clean[20, 20]), np.abs(fshift[20, 20]) * 0.1)

    def test_fft_solid_image_no_false_positive_or_blackout(self):
        white = np.full((128, 128), 255, dtype=np.uint8)
        det_w = process_fft(white, "Balanced", "detect")
        self.assertFalse(det_w["detected"])
        self.assertEqual(det_w["score"], 0.0)
        clean_w = process_fft(white, "Balanced", "clean")
        self.assertTrue(np.array_equal(clean_w, white))

        black = np.zeros((128, 128), dtype=np.uint8)
        det_b = process_fft(black, "Balanced", "detect")
        self.assertFalse(det_b["detected"])
        clean_b = process_fft(black, "Balanced", "clean")
        self.assertTrue(np.array_equal(clean_b, black))

    def test_dct_detection_and_deblocking(self):
        canvas = np.full((128, 128), 255, dtype=np.uint8)
        cv2.line(canvas, (10, 20), (100, 20), 0, 2)
        cv2.line(canvas, (20, 10), (20, 100), 0, 2)
        cv2.putText(canvas, "Code", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)

        det = process_dct(canvas, "Balanced", "detect")
        self.assertFalse(det["detected"])

        rng = random.Random(42)
        noise = np.array([rng.choice([-2, 2]) for _ in range(128 * 128)], dtype=np.int16).reshape((128, 128))
        canvas_wm = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        det_wm = process_dct(canvas_wm, "Balanced", "detect")
        self.assertTrue(det_wm["detected"])

        deblocked = _deblock_8x8(canvas)
        self.assertEqual(deblocked.shape, canvas.shape)
        self.assertEqual(deblocked.dtype, canvas.dtype)

    def test_dwt_text_protection_symmetric(self):
        canvas = np.full((128, 128), 255, dtype=np.uint8)
        cv2.putText(canvas, "Hello", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

        cleaned = process_dwt(canvas, "Balanced", "clean")
        self.assertLess(np.min(cleaned), 30)
        rmse = float(np.sqrt(np.mean((canvas.astype(np.float32) - cleaned.astype(np.float32)) ** 2)))
        self.assertLess(rmse, 2.0, f"DWT asymmetric edge distortion too high: RMSE={rmse}")

    def test_ui_grading_thresholds(self):
        self.assertEqual(get_detection_level("DCT", 73.18, False), ("success", "Clean"))
        self.assertEqual(get_detection_level("DCT", 567.16, True), ("error", "Found"))
        self.assertEqual(get_detection_level("DCT", 300.0, False), ("warning", "Suspicious"))
        self.assertEqual(get_detection_level("FFT", 81.11, False), ("success", "Clean"))
        self.assertEqual(get_detection_level("FFT", 60.0, True), ("error", "Found"))

    def test_single_channel_lsb_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_in = os.path.join(tmpdir, "lsb_single_channel.png")
            logo_path = get_resource_path("logo.png")
            if os.path.exists(logo_path):
                base_img = imread_unicode(logo_path)
            else:
                y, x = np.mgrid[0:256, 0:256]
                base_ch = ((x + y) // 2).astype(np.uint8)
                base_img = cv2.merge([base_ch, base_ch, base_ch])

            wm_img = base_img.copy()
            rng = random.Random(999)
            h, w = wm_img.shape[:2]
            payload = np.array([rng.randint(0, 1) for _ in range(h * w)], dtype=np.uint8).reshape((h, w))
            wm_img[:, :, 0] = (wm_img[:, :, 0] & 0xFE) | payload
            imwrite_unicode(test_in, wm_img)

            processor = WatermarkProcessor()
            res, ok = processor.process_image(
                image_path=test_in,
                methods=["LSB"],
                action="detect",
                color_preserve=True,
                aggressiveness="Balanced"
            )
            self.assertTrue(ok, f"Detection failed: {res}")
            self.assertTrue(res["LSB"]["detected"])
            self.assertGreaterEqual(res["LSB"]["score"], 0.49)

    def test_uint16_png_preservation(self):
        rng = np.random.default_rng(7)
        u16 = rng.integers(25000, 45000, size=(64, 64, 3), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "img16.png")
            out_path = os.path.join(tmpdir, "out16.png")
            self.assertTrue(imwrite_unicode(in_path, u16))

            res = WatermarkProcessor().process_image(
                image_path=in_path,
                methods=["LSB", "FFT", "DCT", "DWT"],
                action="clean",
                color_preserve=True,
                aggressiveness="Balanced",
                output_path=out_path
            )
            self.assertTrue(res.success, res.message)
            cleaned16 = imread_unicode(out_path)
            self.assertEqual(cleaned16.dtype, np.uint16)
            self.assertGreater(int(np.mean(cleaned16)), 20000)
            self.assertTrue(np.all((cleaned16 & 1) == 0))

    def test_privacy_options_and_unicode_paths(self):
        rgba = np.full((64, 64, 4), 200, dtype=np.uint8)
        rgba[:32, :32, 3] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            sub_dir = os.path.join(tmpdir, "中文测试目录")
            os.makedirs(sub_dir, exist_ok=True)
            in_path = os.path.join(sub_dir, "测试截图.png")
            self.assertTrue(imwrite_unicode(in_path, rgba))

            res = WatermarkProcessor().process_image(
                image_path=in_path,
                methods=["LSB", "DCT"],
                action="clean",
                color_preserve=True,
                aggressiveness="Balanced",
                output_path=sub_dir,
                rand_name=True,
                rand_size=True,
                force_jpeg=True
            )
            self.assertTrue(res.success, res.message)
            self.assertTrue(res.output_path.endswith(".jpg"))
            self.assertTrue(os.path.exists(res.output_path))
            out_img = imread_unicode(res.output_path)
            self.assertEqual(out_img.shape[2], 3)
            self.assertGreaterEqual(out_img.shape[0], 60)
            self.assertLessEqual(out_img.shape[0], 64)

            bad_ext = WatermarkProcessor().process_image(
                image_path=in_path,
                methods=["LSB"],
                action="clean",
                color_preserve=True,
                aggressiveness="Balanced",
                output_path=os.path.join(tmpdir, "output.v1")
            )
            self.assertFalse(bad_ext.success)
            self.assertIn("Unsupported output extension", bad_ext.message)

    def test_cancellation_and_pixel_limit_guard(self):
        img = _make_dummy_image(64, 64, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "cancel_test.png")
            self.assertTrue(imwrite_unicode(in_path, img))

            res_cancel = WatermarkProcessor().process_image(
                image_path=in_path,
                methods=["LSB", "FFT"],
                action="detect",
                color_preserve=True,
                aggressiveness="Balanced",
                cancel_check=lambda: True
            )
            self.assertFalse(res_cancel.success)
            self.assertTrue(res_cancel.cancelled)

            with patch.object(processor_mod, "MAX_IMAGE_PIXELS", 1000):
                loaded, err = load_image_checked(in_path)
                self.assertIsNone(loaded)
                self.assertIn("pixel", err.lower())

    def test_exif_orientation_transposition(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "portrait_exif.jpg")
            img = Image.new("RGB", (100, 200), color="blue")
            exif = img.getexif()
            exif[0x0112] = 6  # 90 degrees CW rotation
            img.save(img_path, exif=exif)

            loaded = imread_unicode(img_path)
            self.assertIsNotNone(loaded)
            # 100x200 with orientation 6 should be transposed to 200x100
            self.assertEqual(loaded.shape[0], 100)  # height
            self.assertEqual(loaded.shape[1], 200)  # width

    def test_rand_size_guaranteed_crop(self):
        img = _make_dummy_image(80, 80, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "rand_size_in.png")
            out_path = os.path.join(tmpdir, "rand_size_out.png")
            self.assertTrue(imwrite_unicode(in_path, img))

            # Test 20 times to confirm that 0-crop is strictly never generated
            for _ in range(20):
                res = WatermarkProcessor().process_image(
                    image_path=in_path,
                    methods=["LSB"],
                    action="clean",
                    color_preserve=True,
                    aggressiveness="Balanced",
                    output_path=out_path,
                    rand_size=True
                )
                self.assertTrue(res.success)
                out_img = imread_unicode(out_path)
                self.assertTrue(out_img.shape[0] < 80 or out_img.shape[1] < 80, "Rand Size failed to crop!")
                self.assertGreaterEqual(out_img.shape[0], 76)
                self.assertGreaterEqual(out_img.shape[1], 76)

    def test_dct_gradient_noise_conservative_mode(self):
        rng = np.random.default_rng(42)
        y, x = np.mgrid[0:128, 0:128]
        grad = (x * (200.0 / 128.0)).astype(np.float32)
        noise = rng.normal(0, 3.0, (128, 128)).astype(np.float32)
        img = np.clip(grad + noise, 0, 255).astype(np.uint8)

        det = process_dct(img, "Conservative", "detect")
        self.assertFalse(det["detected"], "Conservative mode falsely detected smooth gradient noise as watermark!")

    def test_uint16_gradient_banding_protection(self):
        ramp = np.linspace(20000, 40000, 64 * 64, dtype=np.float32).reshape((64, 64)).astype(np.uint16)
        cleaned = process_dct(ramp, "Balanced", "clean")
        unique_count = len(np.unique(cleaned))
        self.assertGreater(unique_count, 3000, f"16-bit dynamic range was crushed: {unique_count} unique values")

    def test_new_01_ui_dct_conservative_grading(self):
        rng = np.random.default_rng(42)
        y, x = np.mgrid[0:128, 0:128]
        grad = (x * (200.0 / 128.0)).astype(np.float32)

        # 1. Smooth gradient with subtle natural noise should be completely Clean
        noise_subtle = rng.normal(0, 0.5, (128, 128)).astype(np.float32)
        img_clean = np.clip(grad + noise_subtle, 0, 255).astype(np.uint8)

        # 2. Gradient with moderate noise where score exceeds 200 but detected is False
        noise_mod = rng.normal(0, 3.0, (128, 128)).astype(np.float32)
        img_noise = np.clip(grad + noise_mod, 0, 255).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            p_clean = os.path.join(tmpdir, "grad_clean.png")
            self.assertTrue(imwrite_unicode(p_clean, img_clean))
            res_c, ok_c = WatermarkProcessor().process_image(
                image_path=p_clean,
                methods=["DCT"],
                action="detect",
                color_preserve=True,
                aggressiveness="Conservative"
            )
            self.assertTrue(ok_c)
            self.assertFalse(res_c["DCT"]["detected"])
            lvl_c, lbl_c = get_detection_level("DCT", res_c["DCT"]["score"], res_c["DCT"]["detected"])
            self.assertEqual(lvl_c, "success")
            self.assertEqual(lbl_c, "Clean")

            p_noise = os.path.join(tmpdir, "grad_noise.png")
            self.assertTrue(imwrite_unicode(p_noise, img_noise))
            res_n, ok_n = WatermarkProcessor().process_image(
                image_path=p_noise,
                methods=["DCT"],
                action="detect",
                color_preserve=True,
                aggressiveness="Conservative"
            )
            self.assertTrue(ok_n)
            self.assertFalse(res_n["DCT"]["detected"])
            lvl_n, lbl_n = get_detection_level("DCT", res_n["DCT"]["score"], res_n["DCT"]["detected"])
            # Crucial assertion for NEW-01: UI level must NEVER be "Found" (error) when detected is False
            self.assertNotEqual(lvl_n, "error")
            self.assertNotEqual(lbl_n, "Found")

    def test_new_02_extensionless_path_contract_consistency(self):
        img = _make_dummy_image(64, 64, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_in = os.path.join(tmpdir, "raw_no_ext_input")
            self.assertTrue(imwrite_unicode(raw_in + ".png", img))
            os.rename(raw_in + ".png", raw_in)

            out_dir = os.path.join(tmpdir, "out_sub")
            res = WatermarkProcessor().process_image(
                image_path=raw_in,
                methods=["LSB"],
                action="clean",
                color_preserve=True,
                aggressiveness="Balanced",
                output_path=out_dir
            )
            self.assertTrue(res.success)
            self.assertTrue(os.path.exists(res.output_path), f"Output file does not exist: {res.output_path}")
            self.assertTrue(res.output_path.endswith(".png"))
            loaded = imread_unicode(res.output_path)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, (64, 64, 3))

    def test_algo_01_fft_conservative_periodic_texture_resilience(self):
        y, x = np.mgrid[0:256, 0:256]
        img = (128 + 127 * np.sin((x + y) * 2 * np.pi / 16)).astype(np.uint8)
        res = process_fft(img, "Conservative", "detect")
        self.assertFalse(res["detected"], f"Conservative FFT falsely triggered on sinusoidal texture: {res}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
