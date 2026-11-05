"""Optional: publish the class notebook to your own Kaggle account in one command.

Students can open the notebook straight from GitHub with the "Open in Kaggle" link, so you don't need this.
Use it if you'd rather have a public notebook on your profile that students "Copy & Edit".

Setup (once):  pip install kaggle
               Kaggle → Settings → API → "Create New Token", save kaggle.json to ~/.kaggle/
Run:           python tools/kaggle_publish.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLUG = "train-your-own-mini-llm"


def kaggle_username():
    path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    if os.path.exists(path):
        return json.load(open(path))["username"]
    sys.exit("No Kaggle API token found. Kaggle → Settings → API → Create New Token, then save it to ~/.kaggle/kaggle.json")


def main():
    if not shutil.which("kaggle"):
        sys.exit("The kaggle command isn't installed. Run: pip install kaggle")
    user = kaggle_username()
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(os.path.join(ROOT, "notebooks", "train_your_own_llm.ipynb"), folder)
        meta = {
            "id": f"{user}/{SLUG}",
            "title": "Train Your Own Mini-LLM",
            "code_file": "train_your_own_llm.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": False,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        with open(os.path.join(folder, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        subprocess.run(["kaggle", "kernels", "push", "-p", folder], check=True)
    print(f"\nPublished: https://www.kaggle.com/code/{user}/{SLUG}")
    print("Kaggle now runs it once (about 7 minutes). Open the link, check that Accelerator is 'GPU T4 x2',")
    print("and share it. Students click 'Copy & Edit'.")


if __name__ == "__main__":
    main()
