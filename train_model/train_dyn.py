import sys

from train import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--mode" not in argv:
        argv = ["--mode", "dyn"] + argv
    main(argv)
