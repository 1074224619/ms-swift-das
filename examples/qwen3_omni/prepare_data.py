#!/usr/bin/env python3
# Copyright (c) ModelScope Contributors. All rights reserved.
"""把 VeOmni 风格的 LLaVA-Video 标注转换成 ms-swift 可直接加载的格式.

LLaVA-Video 的 `0_30_s_academic_mc_v0_1_qa_processed.json` 用 `<image>`
作为视频占位符 (VeOmni 在内部按视频处理). ms-swift 的模板按媒体类型分发占位符,
`videos` 字段只会替换 `<video>`, 因此需要先做一次 `<image>` -> `<video>` 的改写.

用法:
    python prepare_data.py --data_dir /path/to/VeOmni

输出:
    <data_dir>/video_swift.json
"""
import argparse
import json
import os
import sys


def convert_video_json(data_dir: str) -> str:
    src = os.path.join(data_dir, 'video.json')
    dst = os.path.join(data_dir, 'video_swift.json')
    if not os.path.isfile(src):
        raise FileNotFoundError(f'video.json not found: {src}')
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_replace = 0
    problems = []
    for i, item in enumerate(data):
        videos = item.get('videos') or []
        if len(videos) != 1:
            problems.append(f'[item {i}] expected 1 video, got {len(videos)}')
            continue
        if not os.path.exists(os.path.join(data_dir, videos[0])):
            problems.append(f'[item {i}] missing video: {videos[0]}')
        for message in item.get('conversations', []):
            value = message.get('value')
            if isinstance(value, str) and '<image>' in value:
                num_replace += value.count('<image>')
                message['value'] = value.replace('<image>', '<video>')

    with open(dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    print(f'saved: {dst}')
    print(f'items: {len(data)}, <image> -> <video>: {num_replace}')
    if problems:
        print(f'warnings ({len(problems)}):')
        for problem in problems[:20]:
            print('  ', problem)
    return dst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data_dir', default='VeOmni',
        help='VeOmni 数据集目录 (包含 video.json / coco / academic_source)')
    args = parser.parse_args()
    convert_video_json(args.data_dir)


if __name__ == '__main__':
    sys.exit(main())
