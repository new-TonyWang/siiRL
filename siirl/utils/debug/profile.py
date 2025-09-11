# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025, Infrawaves. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Callable, Optional

import torch
import torch.distributed

from siirl.utils.params import ProfilerArguments


class Profiler:
    def __init__(self, config):
        # note : if we do not set use_profile, it will be set as None, so that all function will be skip
        self.config = config
        self.skip_prof = False
        self.saved = False
        self.prof = None
        self.rank = torch.distributed.get_rank()
        # we need to validate the config before using the profiler
        self._validate()
        if self.config.get("use_profile") and self.rank in self.config.get("profile_ranks", []):
            print(f"[Profiler] Profiler init for rank {self.rank}")

            step_start = self.config.get("step_start", 0)
            step_end = self.config.get("step_end", 1)

            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=max(step_start - 1, 0),
                    warmup=1 if step_start > 0 else 0,
                    active=step_end - step_start,
                    repeat=1,
                ),
                record_shapes=True,
                with_stack=True,
            )

    def _validate(self):
        if self.config.get("use_profile"):
            if self.config.get("profile_ranks") is None:
                print("[WARNING] Profile ranks is not set, default to rank 0")
                self.config["profile_ranks"] = [0]

            step_start = self.config.get("step_start", 0)
            step_end = self.config.get("step_end", 1)

            assert step_start >= 0, "[ERROR] Profile step start must be greater than 0"
            assert step_end >= 0, "[ERROR] Profile step end must be greater than 0"
            assert step_start < step_end, "[ERROR] Profile step start must be less than step end"

    def check(self):
        return self.prof is not None and not self.skip_prof

    def start(self):
        if self.check():
            print(f"[Profiler] started for rank {self.rank}")
            self.prof.start()

    def step(self):
        if self.check():
            self.prof.step()

    def stop(self):
        if self.check():
            print(f"[Profiler] stopped for rank {self.rank}")
            self.prof.stop()

    def save(self):
        if self.prof is not None and not self.saved:
            save_path = self.config.get("save_path")
            if not save_path:
                print("[WARNING] Profiler save_path is not set, will not save trace.")
                self.skip_prof = True
                return

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            step_start = self.config.get("step_start", 0)
            step_end = self.config.get("step_end", 1)

            save_file_name = f"/prof_start_{step_start}_end_{step_end}_rank_{self.rank}.json"
            full_path = save_path + save_file_name
            print(f"[Profiler] Saving trace to {full_path}")
            self.prof.export_chrome_trace(full_path)
            self.skip_prof = True
            self.saved = True

    def stop_and_save(self):
        if self.check():
            self.stop()
            self.save()

    def stop_trace(self):
        if self.check():
            print(f"[Profiler] Trace stopped for rank {self.rank}")
            self.skip_prof = True


def mark_start_range(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    """Start a profiling range marker (no-op implementation).

    Args:
        message (Optional[str]): Message to associate with the range marker.
        color (Optional[str]): Color for the marker visualization.
        domain (Optional[str]): Domain for the marker.
        category (Optional[str]): Category for the marker.
    """
    pass


def mark_end_range(range_id: str) -> None:
    """End a profiling range marker (no-op implementation).

    Args:
        range_id (str): Identifier of the range to end.
    """
    pass


def mark_annotate(
    message: Optional[str] = None,
    color: Optional[str] = None,
    domain: Optional[str] = None,
    category: Optional[str] = None,
) -> Callable:
    """Decorator to annotate a function with profiling markers (no-op implementation).

    Args:
        message (Optional[str]): Message to associate with the annotation.
        color (Optional[str]): Color for the marker visualization.
        domain (Optional[str]): Domain for the marker.
        category (Optional[str]): Category for the marker.

    Returns:
        Callable: Decorator function that returns the original function unchanged.
    """

    def decorator(func):
        return func

    return decorator


class DistProfiler:

    def __init__(self, rank: int, config: ProfilerArguments, **kwargs):
        pass

    def start(self, **kwargs):
        pass

    def stop(self):
        pass

    @staticmethod
    def annotate(
        message: Optional[str] = None,
        color: Optional[str] = None,
        domain: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs,
    ) -> Callable:
        def decorator(func):
            return func

        return decorator


class DistProfilerExtension:
    """An extension class for DistProfiler that provides distributed profiling capabilities.

    It is intended for workers in verl that single controller invokes.

    This class wraps a DistProfiler instance and provides methods to start/stop profiling
    that can be dispatched across multiple ranks in a distributed training environment.

    Args:
        profiler (DistProfiler): The base distributed profiler instance to extend
    """

    def __init__(self, profiler: DistProfiler):
        self.profiler = profiler

    # from verl.single_controller.base.decorator import Dispatch, register

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    # @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()