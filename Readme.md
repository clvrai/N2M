# N2M: Bridging Navigation and Manipulation by Learning Initial Pose Preference from Rollout
🚧 full code will be released soon!

# installation
we use `uv` to manage our environment. click [here](https://docs.astral.sh/uv/) to get tutorial if you are not familiar with this tool.
```
git clone --single-branch --branch main --recurse-submodules https://github.com/clvrai/N2M.git

cd N2M
uv python install 3.10
uv venv --python=3.10
source .venv/bin/activate

git clone https://github.com/ARISE-Initiative/robosuite
cd robosuite
uv pip install -e .

cd ..
git clone --single-branch https://github.com/cckaixin/robocasa.git
cd robocasa
uv pip install -e .
python robocasa/scripts/download_kitchen_assets.py
python robocasa/scripts/setup_macros.py

cd ..
git clone --single-branch --branch robocasa https://github.com/cckaixin/robomimic.git
cd robomimic
uv pip install -e .
```

## TODO: I will arrange these submodules in a better way. [kaixin]