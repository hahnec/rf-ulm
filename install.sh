# git submodules
git submodule update --init --recursive
ln -sf ./simple_tracker_repo/simple_tracker ./simple_tracker

# virtual environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
