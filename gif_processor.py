"""GIF 处理核心模块。

提供 GIF 帧提取、宫格拼接、自动宫格数选择等功能。
依赖：Pillow
"""

import os
import math
import uuid
from typing import Tuple

from PIL import Image as PILImage


class GifProcessor:
    """GIF 动图处理器。"""

    # 自动选择宫格的时长阈值（秒）和推荐宫格数
    # 规则：时长越短，宫格越少，避免过度采样
    AUTO_GRID_RULES = [
        (0.0, 4),    # 0~1.5s: 4宫格（极短动图）
        (1.5, 4),    # 1.5~3s: 4宫格
        (3.0, 9),    # 3~6s: 9宫格
        (6.0, 9),    # 6~10s: 9宫格
        (10.0, 16),  # 10~18s: 16宫格
        (18.0, 16),  # 18~28s: 16宫格
        (28.0, 25),  # 28s+: 25宫格
    ]

    # 默认最大输出尺寸限制（单张宫格图的长边像素）
    DEFAULT_MAX_OUTPUT_SIZE = 1536

    @staticmethod
    def is_gif(path: str) -> bool:
        """判断文件是否为 GIF 动图。

        仅通过文件头 magic bytes 判断，不检查后缀。
        这是因为平台适配器下载的临时文件可能使用无后缀或任意后缀的命名。
        """
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                header = f.read(6)
                return header in (b"GIF87a", b"GIF89a")
        except OSError:
            return False

    @classmethod
    def auto_grid_size(cls, duration_s: float, frame_count: int) -> int:
        """根据 GIF 时长和帧数智能选择宫格数。

        优先按时长选择，同时确保宫格数不超过帧数（避免重复帧）。
        """
        if frame_count <= 1:
            return 1

        # 按时长找推荐值
        recommended = 4
        for threshold, grid in cls.AUTO_GRID_RULES:
            if duration_s >= threshold:
                recommended = grid

        # 不能超过实际帧数（至少留1帧余量）
        max_grid = max(1, frame_count)
        if recommended > max_grid:
            # 降级到最接近且不超过帧数的完全平方数
            sqrt_val = int(math.isqrt(max_grid))
            recommended = sqrt_val * sqrt_val

        return recommended

    @classmethod
    def parse_grid_preset(cls, preset: str, duration_s: float, frame_count: int) -> int:
        """解析用户配置的宫格预设。"""
        preset = str(preset).strip().lower()
        mapping = {
            "4": 4,
            "4宫格": 4,
            "9": 9,
            "9宫格": 9,
            "16": 16,
            "16宫格": 16,
            "25": 25,
            "25宫格": 25,
            "auto": 0,
            "自动": 0,
        }
        grid = mapping.get(preset, 0)
        if grid == 0:
            grid = cls.auto_grid_size(duration_s, frame_count)
        return grid

    @classmethod
    async def process_gif(
        cls,
        gif_path: str,
        grid_preset: str = "auto",
        cache_dir: str = "",
        max_output_size: int = 0,
    ) -> Tuple[str, dict]:
        """处理 GIF 文件，返回宫格图路径和信息字典。

        Args:
            gif_path: GIF 文件本地路径
            grid_preset: 宫格预设（4/9/16/25/auto/自动）
            cache_dir: 缓存目录
            max_output_size: 宫格图长边最大像素，0 表示使用默认值 1536

        Returns:
            (output_path, info_dict)
        """
        with PILImage.open(gif_path) as im:
            n_frames = getattr(im, "n_frames", 1)

            # 获取每帧延迟，计算总时长
            total_duration_ms = 0
            frame_delays = []
            for i in range(n_frames):
                im.seek(i)
                delay = im.info.get("duration", 100)
                if delay is None or delay <= 0:
                    delay = 100
                frame_delays.append(delay)
                total_duration_ms += delay

            duration_s = total_duration_ms / 1000.0

            # 确定宫格数
            grid_size = cls.parse_grid_preset(grid_preset, duration_s, n_frames)
            grid_side = int(math.isqrt(grid_size))

            # 均匀取帧索引
            if n_frames <= grid_size:
                indices = list(range(n_frames))
                # 如果帧数不足，用最后一帧填充
                while len(indices) < grid_size:
                    indices.append(n_frames - 1)
            else:
                indices = [
                    int(i * (n_frames - 1) / (grid_size - 1))
                    for i in range(grid_size)
                ]

            # 提取帧并转为 RGB
            frames = []
            for idx in indices:
                im.seek(idx)
                frame = im.copy()
                # 处理透明背景：合成到白色背景上
                if frame.mode in ("RGBA", "P"):
                    if frame.mode == "P":
                        frame = frame.convert("RGBA")
                    bg = PILImage.new("RGBA", frame.size, (255, 255, 255, 255))
                    frame = PILImage.alpha_composite(bg, frame).convert("RGB")
                else:
                    frame = frame.convert("RGB")
                frames.append(frame)

        if not frames:
            raise ValueError("未能提取到任何帧")

        # 计算单帧缩放比例，使最终宫格图长边不超过限制
        single_w, single_h = frames[0].size
        total_w = single_w * grid_side
        total_h = single_h * grid_side
        max_total = max_output_size if max_output_size > 0 else cls.DEFAULT_MAX_OUTPUT_SIZE
        scale = 1.0
        if total_w > max_total or total_h > max_total:
            scale = min(max_total / total_w, max_total / total_h)

        new_w = max(1, int(single_w * scale))
        new_h = max(1, int(single_h * scale))

        if scale < 1.0:
            frames = [
                f.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
                for f in frames
            ]

        # 拼接宫格
        grid_img = PILImage.new("RGB", (new_w * grid_side, new_h * grid_side), (255, 255, 255))
        for i, frame in enumerate(frames):
            x = (i % grid_side) * new_w
            y = (i // grid_side) * new_h
            grid_img.paste(frame, (x, y))

        # 保存到缓存目录
        if not cache_dir:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "astrbot_plugin_read_gif")
        os.makedirs(cache_dir, exist_ok=True)

        out_name = f"gif_grid_{uuid.uuid4().hex[:12]}.png"
        out_path = os.path.join(cache_dir, out_name)
        grid_img.save(out_path, "PNG", optimize=True)

        info = {
            "frame_count": n_frames,
            "duration_s": duration_s,
            "grid_size": grid_size,
            "grid_side": grid_side,
            "output_size": grid_img.size,
            "output_path": out_path,
        }
        return out_path, info
