import os

current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)

# Code root: the tree that holds env/, task/, utils/ and env_cfg/. A packaged
# distribution keeps the code inside the wheel, so ROBODOJO_ROOT lets the
# imports resolve there while the data below lives on a separate volume;
# unset, this is upstream's `<repo>` and nothing changes.
ROOT_DIR = os.environ.get("ROBODOJO_ROOT") or os.path.join(current_dir, "..")

# Data root: Assets/ is downloaded separately and is far too large to ship
# inside a wheel, so it may point somewhere else than the code.
ASSETS_ROOT = os.environ.get("ROBODOJO_ASSETS_ROOT") or ROOT_DIR

BENCHMARK = "RoboDojo"
ASSETS_PATH = os.path.join(ASSETS_ROOT, "Assets")
OBJECTS_PATH = os.path.join(ASSETS_ROOT, f"Assets/Object/{BENCHMARK}")
ROBOTS_PATH = os.path.join(ASSETS_ROOT, "Assets/Robots")
ENV_CONFIG_PATH = os.path.join(ROOT_DIR, "env_cfg")

ENV_REGEX_NAMESPACE = "{ENV_REGEX_NS}"

BATCH_NUM = 10

USD_PATH = os.environ.get("ROBODOJO_USD_ASSET_PREFIX") or None
