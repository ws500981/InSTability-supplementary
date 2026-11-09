"""Run the figure and table scripts sequentially."""

import argparse
from pathlib import Path
import subprocess
import sys

GROUPS = {
    "main": ["fig2_s1_s17.py", "fig3.py", "fig4_s28_s36.py"],
    "supplementary": ["fig2_s1_s17.py", "supp_s18.py", "supp_s19.py", "supp_s20_s21.py"]
    + [f"supp_s{i}.py" for i in range(22, 28)]
    + ["fig4_s28_s36.py", "supp_s37.py"],
    "tables": [f"table{i}.py" for i in range(3, 10)],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", *GROUPS], default="all")
    args = parser.parse_args()
    scripts = list(dict.fromkeys(sum(GROUPS.values(), []))) if args.group == "all" else GROUPS[args.group]
    for script in scripts:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / script)], check=True)


if __name__ == "__main__":
    main()
