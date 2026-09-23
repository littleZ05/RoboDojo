# Copyright 2026 The RPent Authors.
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

import sys
from types import SimpleNamespace

from robodojo_runtime import bridge


def test_env_close_releases_recording_capture_and_replicator_in_order(monkeypatch):
    events = []

    class Capture:
        tiled_cameras = [  # noqa: RUF012 - per-test fake class
            SimpleNamespace(
                _render_product_path="/camera/render",
                _annotators={
                    "rgb": SimpleNamespace(detach=lambda paths: events.append(("detach", paths)))
                },
            )
        ]

        def destroy(self):
            events.append("capture")

    env = SimpleNamespace(obs_manager=SimpleNamespace(capture_manager=Capture()))
    facade = object.__new__(bridge.NativeEnv)
    facade.env = env
    facade.app = SimpleNamespace(close=lambda: events.append("app"))
    facade._closed = False
    monkeypatch.setattr(
        bridge.NativeEnv,
        "_stop_replicator",
        staticmethod(lambda: events.append("replicator")),
    )
    assert events == []
    facade.close(False)
    facade.close(False)
    assert events == [
        ("detach", ["/camera/render"]),
        "capture",
        "replicator",
        "app",
    ]


def test_shutdown_clears_stale_syntheticdata_handles_before_replicator_ticks(
    monkeypatch,
):
    stale_handles = ["destroyed-camera-graph"]

    def reset(*, usd):
        assert usd is False  # Render products already destroyed their USD graphs.
        stale_handles.clear()

    def tick():
        assert not stale_handles, "Invalid object in Py_Graph"

    rep = SimpleNamespace(
        orchestrator=SimpleNamespace(
            set_capture_on_play=lambda value: tick(),
            stop=tick,
            wait_until_complete=tick,
        )
    )
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(replicator=SimpleNamespace(core=rep)))
    monkeypatch.setitem(sys.modules, "omni.replicator", SimpleNamespace(core=rep))
    monkeypatch.setitem(sys.modules, "omni.replicator.core", rep)
    monkeypatch.setitem(
        sys.modules,
        "omni.syntheticdata",
        SimpleNamespace(SyntheticData=SimpleNamespace(Get=lambda: SimpleNamespace(reset=reset))),
    )
    bridge.NativeEnv._stop_replicator()
