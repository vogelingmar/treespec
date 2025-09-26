/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch torchvision
pip install -e .