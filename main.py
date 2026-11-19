import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", action="store_true")
    args = parser.parse_args()

    script = "exp_thompson_lp_true.py" if args.true else "exp_thompson_lp.py"
    subprocess.run(["zsh", "run_experiment.sh", script], check=True)


if __name__ == "__main__":
    main()
