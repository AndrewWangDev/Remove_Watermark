from dataclasses import dataclass
import os
import secrets
import string
import cv2
import numpy as np
from PIL import Image, ImageOps
from .algorithms import process_lsb, process_fft, process_dct, process_dwt

MAX_IMAGE_FILE_BYTES = 100 * 1024 * 1024  # 100 MB raw file cap
MAX_IMAGE_PIXELS = 40_000_000             # 40 MP decompression cap
SUPPORTED_OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ProcessResult:
    """Structured result for WatermarkProcessor.process_image with tuple-unpack compatibility."""
    success: bool
    action: str
    message: str = ""
    output_path: str = ""
    detections: dict = None
    cancelled: bool = False

    def __iter__(self):
        payload = self.detections if (self.action == "detect" and self.success) else self.message
        yield payload
        yield self.success


def blend_alpha_to_bgr(bgra_img: np.ndarray, bg_color=(255, 255, 255)) -> np.ndarray:
    """
    Composites a 4-channel BGRA image over a solid background color (default white)
    using true Alpha Blending to prevent dark halos and black boundaries in JPEG export.
    """
    if len(bgra_img.shape) != 3 or bgra_img.shape[2] != 4:
        return bgra_img
    orig_dtype = bgra_img.dtype
    scale = 65535.0 if orig_dtype == np.uint16 else 255.0
    bgr = bgra_img[:, :, :3].astype(np.float32)
    alpha = (bgra_img[:, :, 3].astype(np.float32) / scale)[:, :, np.newaxis]
    bg = np.array(bg_color, dtype=np.float32) * (257.0 if orig_dtype == np.uint16 else 1.0)
    blended = bgr * alpha + bg * (1.0 - alpha)
    if orig_dtype == np.uint16:
        return np.clip(np.round(blended), 0, 65535).astype(np.uint16)
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def load_image_checked(filepath: str, flags: int = cv2.IMREAD_UNCHANGED):
    """
    Validate file size and header dimensions (anti-decompression-bomb) before decoding.
    Returns: (image_array | None, error_message)
    """
    try:
        if not filepath or not os.path.isfile(filepath):
            return None, "Error: File does not exist."

        file_size = os.path.getsize(filepath)
        if file_size > MAX_IMAGE_FILE_BYTES:
            return None, f"Error: Image file exceeds {MAX_IMAGE_FILE_BYTES // (1024 * 1024)} MB safety limit."

        prev_max = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        pil_transposed_arr = None
        try:
            with Image.open(filepath) as pil_img:
                w, h = pil_img.size
                if w * h > MAX_IMAGE_PIXELS:
                    return None, f"Error: Image resolution ({w}x{h}) exceeds {MAX_IMAGE_PIXELS:,} pixel safety limit."

                # If EXIF orientation is present and requires rotation/flip, normalize physical orientation
                exif = pil_img.getexif() if hasattr(pil_img, "getexif") else None
                orientation = exif.get(0x0112) if exif else None
                if orientation in (2, 3, 4, 5, 6, 7, 8):
                    t_img = ImageOps.exif_transpose(pil_img)
                    if t_img is not None:
                        mode = t_img.mode
                        if mode == "RGB":
                            pil_transposed_arr = cv2.cvtColor(np.array(t_img), cv2.COLOR_RGB2BGR)
                        elif mode == "RGBA":
                            pil_transposed_arr = cv2.cvtColor(np.array(t_img), cv2.COLOR_RGBA2BGRA)
                        elif mode == "L":
                            pil_transposed_arr = np.array(t_img)
                        elif mode in ("I;16", "I"):
                            pil_transposed_arr = np.array(t_img, dtype=np.uint16)
                        else:
                            pil_transposed_arr = cv2.cvtColor(np.array(t_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Image.DecompressionBombError:
            return None, f"Error: Image exceeds maximum allowed pixel limit ({MAX_IMAGE_PIXELS:,} pixels)."
        except Exception:
            # Non-PIL formats fall through to post-imdecode shape validation
            pass
        finally:
            Image.MAX_IMAGE_PIXELS = prev_max

        if pil_transposed_arr is not None:
            return pil_transposed_arr, ""

        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        if img is None:
            return None, "Error: Could not load image (unsupported format or corrupted file)."

        h, w = img.shape[:2]
        if h * w > MAX_IMAGE_PIXELS:
            return None, f"Error: Image resolution ({w}x{h}) exceeds {MAX_IMAGE_PIXELS:,} pixel safety limit."
        return img, ""
    except Exception as exc:
        return None, f"Error: Failed to load image ({exc})."


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """Read an image supporting Unicode, Chinese, and special character paths."""
    img, _ = load_image_checked(filepath, flags=flags)
    return img


def imwrite_unicode(filepath: str, img: np.ndarray, quality: int = 95) -> bool:
    """Save an image supporting Unicode, Chinese, and special character paths with atomic replacement."""
    tmp_path = None
    try:
        out_dir = os.path.dirname(os.path.abspath(filepath))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            ext = ".png"
            filepath = filepath + ".png"

        params = []
        if ext in [".jpg", ".jpeg"]:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = blend_alpha_to_bgr(img, (255, 255, 255))
            if img.dtype == np.uint16:
                img = np.clip(np.round(img.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)

        success, buf = cv2.imencode(ext, img, params)
        if not success:
            return False

        # Atomic write: write to a temporary file in the same directory, then atomic rename
        tmp_token = secrets.token_hex(4)
        tmp_path = os.path.join(out_dir, f".tmp_{tmp_token}_{os.path.basename(filepath)}")
        with open(tmp_path, 'wb') as f:
            buf.tofile(f)

        os.replace(tmp_path, filepath)
        tmp_path = None
        return True
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return False

def _resolve_output_path(
    image_path: str,
    output_path: str,
    force_jpeg: bool,
    rand_name: bool
) -> str:
    """Resolve a collision-free output path without overwriting source or existing files."""
    out_path_used = output_path.strip() if output_path else ""
    if not out_path_used or out_path_used.startswith("Default"):
        dir_name, file_name = os.path.split(image_path)
        out_path_used = os.path.join(dir_name, "cleaned_" + file_name)
    else:
        _, ext = os.path.splitext(out_path_used)
        is_dir_target = os.path.isdir(out_path_used) or (not ext) or out_path_used.endswith(("/", "\\"))
        if is_dir_target:
            os.makedirs(out_path_used, exist_ok=True)
            file_name = os.path.basename(image_path)
            out_path_used = os.path.join(out_path_used, "cleaned_" + file_name)
        elif ext.lower() not in SUPPORTED_OUTPUT_EXTS:
            raise ValueError(
                f"Unsupported output extension '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_OUTPUT_EXTS))} (append '/' for a directory)."
            )

    base, cur_ext = os.path.splitext(out_path_used)
    if not cur_ext:
        out_path_used = base + (".jpg" if force_jpeg else ".png")
    elif force_jpeg:
        out_path_used = base + ".jpg"

    src_abs = os.path.abspath(image_path)
    if rand_name:
        alphabet = string.ascii_lowercase + string.digits
        base, ext = os.path.splitext(out_path_used)
        out_dir = os.path.dirname(base)
        for _ in range(32):
            rand_str = ''.join(secrets.choice(alphabet) for _ in range(12))
            candidate = os.path.join(out_dir, f"{rand_str}{ext}")
            if not os.path.exists(candidate) and os.path.abspath(candidate) != src_abs:
                out_path_used = candidate
                break
        else:
            raise RuntimeError("Failed to generate a unique random output filename.")
    else:
        if os.path.exists(out_path_used) or os.path.abspath(out_path_used) == src_abs:
            base, ext = os.path.splitext(out_path_used)
            counter = 1
            while os.path.exists(f"{base}_{counter}{ext}") or os.path.abspath(f"{base}_{counter}{ext}") == src_abs:
                counter += 1
            out_path_used = f"{base}_{counter}{ext}"

    return out_path_used


class WatermarkProcessor:
    def __init__(self):
        pass

    def process_image(
        self,
        image_path: str,
        methods: list,
        action: str,
        color_preserve: bool,
        aggressiveness: str,
        output_path: str = "",
        cancel_check=None,
        rand_name=False,
        rand_size=False,
        force_jpeg=False,
        progress_callback=None
    ) -> ProcessResult:
        """
        Main entry point for processing an image.
        Returns ProcessResult (iterable as (payload, success) for backward compatibility).
        """
        try:
            def _report(pct: int, msg: str = ""):
                if progress_callback:
                    progress_callback(pct, msg)

            _report(5, "Loading image...")
            raw_img, load_err = load_image_checked(image_path, cv2.IMREAD_UNCHANGED)
            if raw_img is None:
                return ProcessResult(success=False, action=action, message=load_err)

            if raw_img.dtype not in (np.uint8, np.uint16):
                return ProcessResult(
                    success=False,
                    action=action,
                    message=f"Error: Unsupported pixel dtype '{raw_img.dtype}'. Only 8-bit and 16-bit images are supported."
                )

            if len(raw_img.shape) not in (2, 3) or (len(raw_img.shape) == 3 and raw_img.shape[2] not in (3, 4)):
                return ProcessResult(
                    success=False,
                    action=action,
                    message=f"Error: Unsupported image channel layout {raw_img.shape}."
                )

            # Extract Alpha channel if present
            alpha_channel = None
            if len(raw_img.shape) == 3 and raw_img.shape[2] == 4:
                alpha_channel = raw_img[:, :, 3]
                bgr_img = raw_img[:, :, :3]
            elif len(raw_img.shape) == 2:
                bgr_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
            else:
                bgr_img = raw_img

            if color_preserve:
                ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
                y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
                channel_to_process = y_channel
            else:
                channel_to_process = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

            results = {}
            current_channel = channel_to_process.copy()
            _report(15, "Initializing processing pipeline...")

            # ----------------------------------------------------
            # DETECTION FLOW
            # ----------------------------------------------------
            if action == "detect":
                algo_map = {
                    "LSB": process_lsb,
                    "FFT": process_fft,
                    "DCT": process_dct,
                    "DWT": process_dwt
                }
                total_methods = len(methods)
                for idx, m_name in enumerate(methods):
                    if cancel_check and cancel_check():
                        return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)

                    step_pct = 20 + int((idx / max(1, total_methods)) * 75)
                    _report(step_pct, f"Running {m_name} detection...")

                    fn = algo_map.get(m_name)
                    if not fn:
                        continue

                    if m_name == "LSB":
                        channels_to_test = []
                        if len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
                            channels_to_test.extend([bgr_img[:, :, 0], bgr_img[:, :, 1], bgr_img[:, :, 2]])
                        else:
                            channels_to_test.append(current_channel)

                        if alpha_channel is not None:
                            channels_to_test.append(alpha_channel)

                        best_detected = False
                        max_score = 0.0
                        for ch_data in channels_to_test:
                            if cancel_check and cancel_check():
                                return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)
                            ch_res = fn(ch_data, aggressiveness, "detect", cancel_check=cancel_check)
                            if ch_res is None:
                                return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)
                            if ch_res.get("detected"):
                                best_detected = True
                            max_score = max(max_score, ch_res.get("score", 0.0))

                        det_res = {"detected": bool(best_detected), "score": float(round(max_score, 4))}
                    else:
                        det_res = fn(current_channel, aggressiveness, "detect", cancel_check=cancel_check)
                        if det_res is None:
                            return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)

                    results[m_name] = det_res

                _report(100, "Detection complete")
                if not results:
                    return ProcessResult(success=False, action=action, message="Error: Please select at least one valid detection method.")
                return ProcessResult(success=True, action=action, detections=results)

            # ----------------------------------------------------
            # CLEANING FLOW (Frequency Filtering -> Color Reconstruction -> Final LSB Sweep)
            # ----------------------------------------------------
            if not methods:
                return ProcessResult(success=False, action=action, message="Error: No clean method selected.")

            freq_methods = [m for m in ["FFT", "DCT", "DWT"] if m in methods]
            freq_algo_map = {
                "FFT": process_fft,
                "DCT": process_dct,
                "DWT": process_dwt
            }

            for idx, m_name in enumerate(freq_methods):
                if cancel_check and cancel_check():
                    return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)
                step_pct = 20 + int((idx / max(1, len(freq_methods) + 1)) * 50)
                _report(step_pct, f"Filtering {m_name} artifacts...")

                fn = freq_algo_map[m_name]
                current_channel = fn(current_channel, aggressiveness, "clean", cancel_check=cancel_check)
                if current_channel is None:
                    return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)

            if cancel_check and cancel_check():
                return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)

            _report(75, "Reconstructing color spaces...")
            if color_preserve:
                if aggressiveness == "Aggressive":
                    cr_channel = cv2.GaussianBlur(cr_channel, (3, 3), 0)
                    cb_channel = cv2.GaussianBlur(cb_channel, (3, 3), 0)
                elif aggressiveness == "Balanced":
                    cr_channel = cv2.medianBlur(cr_channel, 3)
                    cb_channel = cv2.medianBlur(cb_channel, 3)

                merged = cv2.merge([current_channel, cr_channel, cb_channel])
                result_img = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
            else:
                result_img = current_channel

            if alpha_channel is not None:
                if force_jpeg:
                    if len(result_img.shape) == 2:
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                    bgra_combined = cv2.merge([result_img[:, :, 0], result_img[:, :, 1], result_img[:, :, 2], alpha_channel])
                    result_img = blend_alpha_to_bgr(bgra_combined, (255, 255, 255))
                else:
                    if len(result_img.shape) == 2:
                        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
                    result_img = cv2.merge([result_img[:, :, 0], result_img[:, :, 1], result_img[:, :, 2], alpha_channel])

            if "LSB" in methods:
                if cancel_check and cancel_check():
                    return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)
                _report(85, "Neutralizing spatial LSB planes...")
                if len(result_img.shape) == 3:
                    channels = list(cv2.split(result_img))
                    for c_idx in range(len(channels)):
                        cleaned_ch = process_lsb(channels[c_idx], aggressiveness, "clean", cancel_check=cancel_check)
                        if cleaned_ch is None:
                            return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)
                        channels[c_idx] = cleaned_ch
                    result_img = cv2.merge(channels)
                else:
                    result_img = process_lsb(result_img, aggressiveness, "clean", cancel_check=cancel_check)
                    if result_img is None:
                        return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)

            if cancel_check and cancel_check():
                return ProcessResult(success=False, action=action, message="Task cancelled by user.", cancelled=True)

            if rand_size:
                h, w = result_img.shape[:2]
                crop_t, crop_b, crop_l, crop_r = 0, 0, 0, 0
                while (crop_t + crop_b + crop_l + crop_r) == 0:
                    crop_t = secrets.randbelow(3)
                    crop_b = secrets.randbelow(3)
                    crop_l = secrets.randbelow(3)
                    crop_r = secrets.randbelow(3)
                if h - crop_t - crop_b > 10 and w - crop_l - crop_r > 10:
                    result_img = result_img[crop_t:h - crop_b, crop_l:w - crop_r]

            out_path_used = _resolve_output_path(image_path, output_path, force_jpeg, rand_name)

            _report(94, "Writing output image...")
            save_ok = imwrite_unicode(out_path_used, result_img)
            if not save_ok:
                raise RuntimeError(f"Failed to write image to [{out_path_used}]. Check directory write permissions.")

            _report(100, "Clean complete")
            return ProcessResult(
                success=True,
                action=action,
                message=f"Image cleaned and saved to:\n{out_path_used}",
                output_path=out_path_used
            )

        except Exception as e:
            import traceback
            return ProcessResult(
                success=False,
                action=action,
                message=f"Error during processing: {str(e)}\n{traceback.format_exc()}"
            )
