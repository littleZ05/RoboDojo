# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboDojo environment runtime.

RoboDojo's Python code is a source tree, not an importable distribution: its
top-level directories (``env``, ``task``, ``utils``) rely on being on
``PYTHONPATH``, and the package runs inside Isaac Sim. This distribution makes
that tree installable without changing how RoboDojo imports itself:

* :data:`VENDORED_SOURCE_ROOT` holds the validated, patched RoboDojo tree and
  :func:`source_root` returns it, so callers can put it on ``PYTHONPATH`` and
  ``import env.global_configs`` exactly as they would from a checkout.
* the simulator dependency pins ride along in this distribution instead of in
  the integrator's extras, where they collided with the agent stack.

Scene data (``Assets/``) and policy checkpoints are *not* shipped: they are
downloaded separately. :func:`assets_root` reports where the patched
``env.global_configs`` looks for them.

The vendored changes are recorded in ``SOURCE_LOCK.txt``.
``env/global_configs.py`` accepts ``ROBODOJO_ROOT`` for the code root and
``ROBODOJO_ASSETS_ROOT`` for the scene-data root. Unset, both keep upstream's
``<repo>/Assets`` behaviour, so a checkout and this distribution behave
identically.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "VENDORED_SOURCE_ROOT",
    "assets_root",
    "env_config_path",
    "source_root",
    "__version__",
]

__version__ = "0.3.0"

_PACKAGE_DIR = Path(__file__).resolve().parent

#: Upstream RoboDojo tree vendored next to this package.
VENDORED_SOURCE_ROOT = _PACKAGE_DIR / "_source"


def source_root() -> Path:
    """Return the RoboDojo source root to put on ``PYTHONPATH``.

    ``ROBODOJO_SOURCE_ROOT`` wins when set, so a developer checkout can be used
    without reinstalling. The directory holds ``env/``, ``src/``, ``task/``,
    ``utils/`` and ``env_cfg/``, the same layout as the upstream repository
    root.
    """
    override = os.environ.get("ROBODOJO_SOURCE_ROOT")
    root = Path(override).expanduser() if override else VENDORED_SOURCE_ROOT
    return root.resolve()


def assets_root() -> Path:
    """Return the directory that holds RoboDojo scene data.

    Mirrors ``ROBODOJO_ASSETS_ROOT`` in the patched ``env.global_configs``.
    RoboDojo reads ``<assets root>/Assets``; the data is downloaded separately,
    so this reports the destination to link or download into, not a promise
    that the files are present.
    """
    override = os.environ.get("ROBODOJO_ASSETS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return source_root()


def env_config_path() -> Path:
    """Return the directory holding RoboDojo's ``env_cfg/*.yml`` configurations."""
    return source_root() / "env_cfg"
