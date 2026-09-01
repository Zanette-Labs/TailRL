"""VENDORED SE-GUI reference reward -- the ground truth our implementation is tested against.

Source: https://github.com/YXB-NKU/SE-GUI
        src/open-r1-multimodal/src/open_r1/vlm_modules/qwen_module.py
        Qwen2VLModule.point_reward (defined at ~line 97)
Fetched: 2026-08-03, sha256(file) = 844d136ec12a8a2b62eb9da8001e4cb5c08337c2e9cbf3134d3c3a5fe636c78b
        (identical on refs main and master)

VERBATIM except: the `if os.getenv("DEBUG_MODE") == "true"` file-logging block and its `os` /
`datetime` imports and `current_time` variable are stripped (they write to LOG_PATH and have no
effect on the returned reward). Nothing else is touched -- not the arithmetic, not the ordering,
not the original comments. DO NOT "clean up" this file: its whole value is being byte-faithful to
the code that produced SE-GUI's released checkpoints.

Note the two upstream quirks this file preserves, both deliberate (see docs/segui/errata.md):
  * decay is 1 - (d/d_max)**2, NOT the paper's (1 - d/d_max)**2;
  * the `if d <= 1` guard is on the RAW normalized distance, not on d/d_max.
"""


def point_reward(completions, solution, **kwargs):
    """Calculate reward based on whether the predicted point is inside the bounding box and its distance from the box center."""
    import re
    import json
    import math

    # 从每个 completion 中提取 content
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    # 遍历每个 content 和对应的 solution
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            # 使用正则表达式提取 <tool_call> 标签中的内容
            tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            if tool_call_match:
                tool_call_content = tool_call_match.group(1).strip()
                # 解析 JSON
                tool_call_json = json.loads(tool_call_content)
                arguments = tool_call_json.get("arguments", {})
                coordinate = arguments.get("coordinate", None)
                # 检查坐标是否是一个长度为 2 的列表
                if coordinate and isinstance(coordinate, list) and len(coordinate) == 2:
                    x, y = coordinate
                    # 确保 x 和 y 是数值类型
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        # 提取边界框和图像尺寸
                        box = sol[:4]  # [x_min, y_min, x_max, y_max]
                        img_width, img_height = sol[4], sol[5]

                        # 检查点是否在边界框内
                        if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                            base_reward = 1.0
                        else:
                            base_reward = 0.0

                        # 计算边界框中心
                        cx = (box[0] + box[2]) / 2
                        cy = (box[1] + box[3]) / 2

                        # 归一化坐标
                        nx = x / img_width
                        ny = y / img_height
                        ncx = cx / img_width
                        ncy = cy / img_height

                        # 计算边界框中心到图像四个角的归一化距离
                        d1 = math.sqrt((ncx - 0)**2 + (ncy - 0)**2)
                        d2 = math.sqrt((ncx - 1)**2 + (ncy - 0)**2)
                        d3 = math.sqrt((ncx - 0)**2 + (ncy - 1)**2)
                        d4 = math.sqrt((ncx - 1)**2 + (ncy - 1)**2)
                        max_d = max(d1, d2, d3, d4)

                        # 计算点到中心的归一化距离
                        d = math.sqrt((nx - ncx)**2 + (ny - ncy)**2)
                        d_normalized = d / max_d if max_d > 0 else 0
                        decay_term = 1 - d_normalized**2 if d <= 1 else 0

                        # 总奖励
                        reward = base_reward + decay_term
        except Exception:
            # 如果解析失败或发生异常，reward 保持为 0.0
            pass

        rewards.append(reward)

    return rewards
