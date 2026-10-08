"""
Prompt 构建工具
用于生成各种先验类型的检测提示词
"""

from typing import List, Tuple, Optional


def build_none_prompt() -> str:
    """构建无先验提示词（baseline）"""

    return (
        "Detect all text regions in this image. "
        "For each text region, output a JSON object with field "
        "\"bbox_2d\": [xmin, ymin, xmax, ymax]. "
        "Output a JSON array of all such objects. "
        "Coordinates are integers in the range [0, 1000]. "
        "Example output for 2 regions:\n"
        "[{\"bbox_2d\": [120, 85, 310, 115]}, "
        "{\"bbox_2d\": [450, 420, 550, 520]}]"
    )


def build_center_prompt(
    center_points: List[Tuple[int, int]],
    max_points: int = 20
) -> str:
    """
    构建中心点先验提示词

    Args:
        center_points: [(x1, y1), (x2, y2), ...] 中心点坐标
        max_points: 最多列举多少个点

    Returns:
        完整的提示词
    """

    # 截断处理
    if len(center_points) > max_points:
        points_to_use = center_points[:max_points]
        hint = f"To help you, the center points of {max_points} prominent text regions are provided here: "
    else:
        points_to_use = center_points
        hint = "The center points of these text regions are: "

    # 格式化坐标字符串
    points_str = ", ".join(f"({x}, {y})" for x, y in points_to_use)

    return (
        "Detect ALL text regions in this image. "
        f"{hint}{points_str}. "
        "For each text region you detect, output a JSON object with field "
        "\"bbox_2d\": [xmin, ymin, xmax, ymax]. "
        "Output a JSON array containing ALL detected objects, "
        "including those without provided points. "
        "Coordinates are integers in the range [0, 1000]. "
        "Example output for 2 regions:\n"
        "[{\"bbox_2d\": [120, 85, 310, 115]}, "
        "{\"bbox_2d\": [450, 420, 550, 520]}]"
    )


def build_text_prompt(
    texts: List[str],
    max_texts: int = 20
) -> str:
    """
    构建文本先验提示词

    Args:
        texts: ["text1", "text2", ...] 文本内容列表
        max_texts: 最多列举多少个文本

    Returns:
        完整的提示词
    """

    # 截断处理
    if len(texts) > max_texts:
        texts_to_use = texts[:max_texts]
        hint = f"To help you, the content of {max_texts} prominent text regions are provided here: "
    else:
        texts_to_use = texts
        hint = "The text contents of these text regions are: "

    # 格式化文本字符串（每个文本用双引号包围）
    texts_str = ", ".join(f'"{t}"' for t in texts_to_use)

    return (
        "Detect ALL text regions in this image. "
        f"{hint}{texts_str}. "
        "For each text region, output a JSON object with field "
        "\"bbox_2d\": [xmin, ymin, xmax, ymax]. "
        "Output a JSON array containing ALL detected objects, "
        "including those without provided text contents. "
        "Coordinates are integers in the range [0, 1000]. "
        "Example output for 2 regions:\n"
        "[{\"bbox_2d\": [120, 85, 310, 115]}, "
        "{\"bbox_2d\": [450, 420, 550, 520]}]"
    )


def build_random_point_prompt(
    random_points: List[Tuple[int, int]],
    max_points: int = 20
) -> str:
    """
    构建随机点先验提示词

    Args:
        random_points: [(x1, y1), (x2, y2), ...] 框内随机点
        max_points: 最多列举多少个点

    Returns:
        完整的提示词
    """

    # 结构与中心点先验相同，仅改变点的来源
    if len(random_points) > max_points:
        points_to_use = random_points[:max_points]
        hint = f"To help you, the random points of {max_points} prominent text regions are provided here: "
    else:
        points_to_use = random_points
        hint = "The random points of these text regions are: "

    points_str = ", ".join(f"({x}, {y})" for x, y in points_to_use)

    return (
        "Detect ALL text regions in this image. "
        f"{hint}{points_str}. "
        "For each text region you detect, output a JSON object with field "
        "\"bbox_2d\": [xmin, ymin, xmax, ymax]. "
        "Output a JSON array containing ALL detected objects. "
        "Coordinates are integers in the range [0, 1000]. "
        "Example output for 2 regions:\n"
        "[{\"bbox_2d\": [120, 85, 310, 115]}, "
        "{\"bbox_2d\": [450, 420, 550, 520]}]"
    )


# 使用示例
if __name__ == "__main__":
    # 无先验
    print("=" * 60)
    print("BASELINE (无先验)")
    print("=" * 60)
    print(build_none_prompt())

    # 中心点先验
    print("\n" + "=" * 60)
    print("中心点先验")
    print("=" * 60)
    centers = [(100, 150), (300, 200), (450, 300)]
    print(build_center_prompt(centers))

    # 文本先验
    print("\n" + "=" * 60)
    print("文本先验")
    print("=" * 60)
    texts = ["Hello", "World", "Sample Text"]
    print(build_text_prompt(texts))
