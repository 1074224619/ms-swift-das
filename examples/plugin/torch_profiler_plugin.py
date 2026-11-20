# Copyright (c) ModelScope Contributors. All rights reserved.
"""ms-swift Megatron 训练流程的 torch.profiler 抓取插件。

用法（无需修改 ms-swift 源码）：

    --external_plugins /cfs/megatron-code/ms-swift/tools/torch_profiler_plugin.py \
    --callbacks torch_profiler

可选环境变量（因为 cli_main 不接受未知参数，配置一律走环境变量）：

    SWIFT_PROF_RANK          只在这一个全局 rank 上抓，默认 0
    SWIFT_PROF_DIR           输出目录，默认 <output_dir>/torch_prof
    SWIFT_PROF_WAIT          跳过前 N 个 step，默认 5   （避开 JIT / 编译 / 显存池预热）
    SWIFT_PROF_WARMUP        预热 N 个 step，默认 3
    SWIFT_PROF_ACTIVE        实际记录 N 个 step，默认 1
    SWIFT_PROF_REPEAT        循环次数，默认 1
    SWIFT_PROF_RECORD_SHAPES 默认 0
    SWIFT_PROF_MEMORY        默认 0（打开可看显存分配，trace 会大很多）
    SWIFT_PROF_WITH_STACK    默认 0（打开后可在 trace 里看 python 调用栈）
    SWIFT_PROF_FLOPS         默认 0

产物（在 SWIFT_PROF_DIR 下）：
    <hostname>_<pid>.*.pt.trace.json   kineto trace，用 Perfetto / chrome://tracing 打开
    chrome_trace.json                  同一份 trace 的副本（固定文件名，便于拉取）
    key_averages.txt                   按 self device time 排序的算子耗时表

原理：ms-swift 的 megatron_callbacks_map 是按名字查表的，插件在被 import 时
往表里注册一个新回调即可，训练循环会在 on_train_begin / on_step_end /
on_train_end 处调用它。
"""

import atexit
import os
import socket

import torch

from swift.megatron.callbacks.base import MegatronCallback
from swift.megatron.callbacks.mapping import megatron_callbacks_map
from swift.utils import get_logger

logger = get_logger()

_TRUE = {'1', 'true', 'True', 'yes', 'y', 'on'}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        logger.warning(f'[torch_profiler] invalid {name}={os.environ.get(name)!r}, fallback to {default}.')
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)) in _TRUE


class TorchProfilerCallback(MegatronCallback):
    """在指定的全局 rank 上按 schedule 抓取若干 step 的 torch profiler trace。"""

    def on_train_begin(self):
        self.prof = None
        self.finished = False
        self.out_dir = None

        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        target_rank = _env_int('SWIFT_PROF_RANK', 0)
        if rank != target_rank:
            return

        args = self.args
        out_dir = os.environ.get('SWIFT_PROF_DIR') or os.path.join(args.output_dir, 'torch_prof')
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

        wait = _env_int('SWIFT_PROF_WAIT', 5)
        warmup = _env_int('SWIFT_PROF_WARMUP', 3)
        active = _env_int('SWIFT_PROF_ACTIVE', 1)
        repeat = _env_int('SWIFT_PROF_REPEAT', 1)

        logger.info(
            f'[torch_profiler] rank={rank} out_dir={out_dir} '
            f'schedule(wait={wait}, warmup={warmup}, active={active}, repeat={repeat}) '
            f'train_iters={args.train_iters}')

        # torch.profiler.ProfilerActivity.CUDA 在 ROCm/HIP 构建下就是 device 侧 activity。
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        self.prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=_env_bool('SWIFT_PROF_RECORD_SHAPES', False),
            profile_memory=_env_bool('SWIFT_PROF_MEMORY', False),
            with_stack=_env_bool('SWIFT_PROF_WITH_STACK', True),
            with_flops=_env_bool('SWIFT_PROF_FLOPS', False),
        )
        self.prof.start()
        # 训练异常退出时 on_train_end 不会触发，用 atexit 兜底，避免 trace 丢失。
        atexit.register(self._finalize)

    def on_step_end(self):
        if self.prof is not None:
            self.prof.step()

    def on_train_end(self):
        self._finalize()

    def _finalize(self):
        if self.prof is None or self.finished:
            return
        self.finished = True

        try:
            self.prof.stop()
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[torch_profiler] profiler.stop() failed: {e}')
            return

        try:
            # 固定的副本文件名，方便直接从跳板机 scp 拉走。
            trace_path = os.path.join(self.out_dir, 'chrome_trace.json')
            self.prof.export_chrome_trace(trace_path)
            logger.info(f'[torch_profiler] chrome trace: {trace_path}')
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[torch_profiler] export_chrome_trace failed: {e}')

        try:
            key_averages = self.prof.key_averages()
            try:
                table = key_averages.table(sort_by='self_cuda_time_total', row_limit=30)
            except Exception:  # noqa: BLE001
                table = key_averages.table(row_limit=30)
            table_path = os.path.join(self.out_dir, 'key_averages.txt')
            with open(table_path, 'w', encoding='utf-8') as f:
                f.write(table)
            logger.info(f'[torch_profiler] key_averages: {table_path}\n{table}')
        except Exception as e:  # noqa: BLE001
            logger.warning(f'[torch_profiler] key_averages failed: {e}')

        logger.info(f'[torch_profiler] done on {socket.gethostname()}, artifacts in {self.out_dir}')


megatron_callbacks_map['torch_profiler'] = TorchProfilerCallback
