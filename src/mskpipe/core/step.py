# SPDX-License-Identifier: Apache-2.0
"""Step interface shared by all pipeline steps."""

from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from mskpipe.config import InputSpec, PipelineConfig
from mskpipe.core.device import DeviceReport
from mskpipe.core.manifest import StepRecord
from mskpipe.core.workspace import Workspace


class StepError(RuntimeError):
    """A step failed in an expected way (bad input, tool error)."""


@dataclass
class StepContext:
    ws: Workspace
    config: PipelineConfig
    input: InputSpec
    record: StepRecord
    step_name: str
    logger: logging.Logger
    device: DeviceReport = field(
        default_factory=lambda: DeviceReport(requested="cpu", selected="cpu", reason="default")
    )

    @property
    def out_dir(self) -> Path:
        """Output folder of the current step."""
        return self.ws.step_dir(self.step_name)

    def step_dir(self, step: str) -> Path:
        """Output folder of another (typically upstream) step."""
        return self.ws.step_dir(step)

    def run_command(
        self,
        args: Sequence[str | Path],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> None:
        """Run an external tool; output goes to ``logs/<step>.log``.

        The tool gets the run's device (``CUDA_VISIBLE_DEVICES``; empty on CPU runs) and
        thread limits; ``env`` adds to or overrides the current environment.
        """
        argv = [str(a) for a in args]
        full_env = {**os.environ, **self.device.env(self.config.runtime.threads), **(env or {})}
        log_path = self.ws.logs_dir / f"{self.step_name}.log"
        self.logger.info("$ %s", subprocess.list2cmdline(argv))
        with (
            log_path.open("a", encoding="utf-8") as log,
            subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=full_env,
                cwd=cwd,
            ) as proc,
        ):
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                self.logger.debug("%s", line.rstrip())
            code = proc.wait()
        if code != 0:
            raise StepError(
                f"'{Path(argv[0]).name}' exited with code {code}; "
                f"see {log_path.relative_to(self.ws.root).as_posix()}"
            )


class Step(ABC):
    """One pipeline step. ``name`` must be a key of ``workspace.STEP_DIRS``."""

    name: ClassVar[str]
    version: ClassVar[str] = "1"  # bump when a code change alters outputs
    config_sections: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def run(self, ctx: StepContext) -> None: ...

    def fingerprint_extra(self, config: PipelineConfig) -> Mapping[str, Any]:
        """Extra identity mixed into the cache key, e.g. the selected plugin and its version."""
        return {}
