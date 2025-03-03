import click
from safetensors.torch import save_file
from safetensors import safe_open
import subprocess

@click.command()
@click.option("--wavdir", default="wav", type=str)
@click.option("--trainpath", default="train-list.txt", type=str)
@click.option("--valpath", default="val-list.txt", type=str)
@click.option("--outpath", default="pitch.safetensors", type=str)
@click.option("--split", default=8, type=int)
def main(wavdir, trainpath, valpath, outpath, split):
    with open(trainpath, "r") as f:
        trainlines = f.readlines()
    with open(valpath, "r") as f:
        vallines = f.readlines()
    lines = trainlines + vallines
    children = []
    hop = len(lines) // split
    begin = 0
    end = hop
    for i in range(split - 1):
        with open(f"tmp-{i}.txt", "w") as f:
            f.write("".join(lines[begin:end]))
        children.append(subprocess.Popen(["python", "stylish-dataset/single-pitch.py", "--wavdir", wavdir, "--inpath", f"tmp-{i}.txt", "--outpath", f"tmp-{i}.safetensors"]))
        begin += hop
        end += hop
    with open(f"tmp-{split-1}.txt", "w") as f:
        f.write("".join(lines[begin:]))
    children.append(subprocess.Popen(["python", "stylish-dataset/single-pitch.py", "--wavdir", wavdir, "--inpath", f"tmp-{split-1}.txt", "--outpath", f"tmp-{split-1}.safetensors"]))
    for child in children:
        child.wait()
    result = {}
    for i in range(split):
        with safe_open(f"tmp-{i}.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                result[key] = f.get_tensor(key)
    save_file(result, outpath)

if __name__ == "__main__":
    main()
