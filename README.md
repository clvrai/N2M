# N2M: Bridging Navigation and Manipulation by Learning Initial Pose Preference from Rollout
Kaixin Chai*, Hyunjun Lee*, Joseph J. Lim

🚧 full code will be released soon!

## TODOs
- Organize N2M training and inference code
- Organize robocasa evironment code
- Clean up README.md file

## Installation
we use `uv` to manage our environment. click [here](https://docs.astral.sh/uv/) to get tutorial if you are not familiar with this tool.
```
git clone --single-branch --branch main --recurse-submodules https://github.com/clvrai/N2M.git

cd N2M
uv python install 3.10
uv venv --python=3.10
source .venv/bin/activate

cd ../robosuite
uv pip install -e .

cd ../robocasa
uv pip install -e .
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py

cd ../robomimic
uv pip install -e .
```