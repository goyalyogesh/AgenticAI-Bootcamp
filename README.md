# AgenticAI-Bootcamp

## Setup Instructions

### Windows (Anaconda Prompt)
Navigate to the current working directory and run the following commands:

```bash
conda create --prefix ./env python=3.12 -y
conda activate ./env
pip install -r requirements.txt
pip install python-dotenv
python main.py
```

## Git Helper Commands

To remove a file from git's index (stop tracking):

```bash
git rm --cached path/to/your/file.txt
git commit -m "Stop tracking and ignore path/to/your/file.txt"
```