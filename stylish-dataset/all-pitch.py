import click
from safetensors.torch import save_file
from safetensors import safe_open
import subprocess


@click.command()
@click.option(
    "--method",
    default="pyworld",
    type=click.Choice(["pyworld", "rmvpe"], case_sensitive=False),
)
@click.option("--wavdir", default="wav", type=str)
@click.option("--trainpath", default="train-list.txt", type=str)
@click.option("--valpath", default="val-list.txt", type=str)
@click.option("--outpath", default="pitch.safetensors", type=str)
@click.option("--split", default=8, type=int)
@click.option("--rmvpe_checkpoint", default=None, type=str)
def main(method, wavdir, trainpath, valpath, outpath, split, rmvpe_checkpoint=None):
    with open(trainpath, "r", encoding="utf-8") as f:
        trainlines = f.readlines()
    with open(valpath, "r", encoding="utf-8") as f:
        vallines = f.readlines()
    lines = trainlines + vallines
    children = []
    hop = len(lines) // split
    begin = 0
    end = hop
    for i in range(split - 1):
        with open(f"tmp-{i}.txt", "w", encoding="utf-8") as f:
            f.write("".join(lines[begin:end]))
        children.append(
            subprocess.Popen(
                [
                    "python",
                    "stylish-dataset/single-pitch.py",
                    "--method",
                    method,
                    "--wavdir",
                    wavdir,
                    "--inpath",
                    f"tmp-{i}.txt",
                    "--outpath",
                    f"tmp-{i}.safetensors",
                    "--process_id",
                    str(i),
                ]
                + (
                    ["--rmvpe_checkpoint", rmvpe_checkpoint]
                    if method == "rmvpe"
                    else []
                )
            )
        )
        begin += hop
        end += hop
    with open(f"tmp-{split-1}.txt", "w") as f:
        f.write("".join(lines[begin:]))
    children.append(
        subprocess.Popen(
            [
                "python",
                "stylish-dataset/single-pitch.py",
                "--method",
                method,
                "--wavdir",
                wavdir,
                "--inpath",
                f"tmp-{split-1}.txt",
                "--outpath",
                f"tmp-{split-1}.safetensors",
                "--process_id",
                str(split - 1),
            ]
            + (["--rmvpe_checkpoint", rmvpe_checkpoint] if method == "rmvpe" else [])
        )
    )
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
