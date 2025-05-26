import pathlib, sys

import click
import numpy
import soundfile
import torch
import librosa

from safetensors.torch import save_file

device = "cuda"


@click.command()
@click.option(
    "--method",
    default="pyworld",
    type=click.Choice(["pyworld", "rmvpe"], case_sensitive=False),
)
@click.option("--wavdir", default="wav", type=str)
@click.option("--inpath", default="list.txt", type=str)
@click.option("--outpath", default="pitch.safetensors", type=str)
@click.option("--rmvpe_checkpoint", default=None, type=str)
@click.option("--process_id", default=0, type=int)
def main(method, wavdir, inpath, outpath, rmvpe_checkpoint=None, process_id=0):
    method = method.lower()
    wavdir = pathlib.Path(wavdir)
    if method == "pyworld":
        result = calculate_pitch_pyworld(pathlib.Path(inpath), wavdir, process_id)
    elif method == "rmvpe":
        assert (
            rmvpe_checkpoint
        ), "Pitch extraction method RVMPE requires a pretrained weight. Specify it with --rmvpe_checkpoint /path/to/pretrained/rmvpe/checkpoint"
        result = calculate_pitch_rmvpe(
            pathlib.Path(inpath), wavdir, rmvpe_checkpoint, process_id
        )
    save_file(result, outpath)


def calculate_pitch_pyworld(path, wavdir, process_id):
    import pyworld

    result = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    for count, line in enumerate(lines, 1):
        fields = line.split("|")
        name = fields[0]
        try:
            wave, sr = soundfile.read(wavdir / name)
            if sr != 24000:
                print(f"Skipping {name}: Wrong sample rate ({sr})")
        except:
            print(f"Skipping {name}: File not found or corrupted")
            continue
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        time_bin = get_time_bin(wave.shape[0])
        if time_bin == -1:
            print(f"Skipping {name}: Too short")
            continue
        frame_count = get_frame_count(time_bin)
        pad_start = (frame_count * 300 - wave.shape[0]) // 2
        pad_end = frame_count * 300 - wave.shape[0] - pad_start
        wave = numpy.concatenate(
            [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
        )

        bad_f0 = 5
        zero_value = -10
        frame_period = 300 / 24000 * 1000
        f0, t = pyworld.harvest(wave, 24000, frame_period=frame_period)
        # if harvest fails, try dio
        if sum(f0 != 0) < bad_f0:
            print("D")
            f0, t = pyworld.dio(wave, 24000, frame_period=frame_period)
        pitch = pyworld.stonemask(wave, f0, t, 24000)
        pitch = torch.from_numpy(pitch).float().unsqueeze(0)
        if torch.any(torch.isnan(pitch)):
            pitch[torch.isnan(pitch)] = zero_value

        result[name] = pitch
        print(".", end=" ")
        if count % 100 == 0:
            print(f"P{process_id} {count}/{len(lines)}")
    return result


def calculate_pitch_rmvpe(path, wavdir, checkpoint, process_id):
    from rmvpe import RMVPE

    rmvpe = RMVPE(checkpoint)
    zero_value = -10
    result = {}
    lines = path.read_text(encoding="utf-8").splitlines()

    for count, line in enumerate(lines, 1):
        fields = line.split("|")
        name = fields[0]
        wave, sr = soundfile.read(wavdir / name)
        try:
            wave, sr = soundfile.read(wavdir / name)
            if sr != 24000:
                print(f"Skipping {name}: Wrong sample rate ({sr})")
        except:
            print(f"Skipping {name}: File not found or corrupted")
            continue
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        time_bin = get_time_bin(wave.shape[0])
        if time_bin == -1:
            print(f"Skipping {name}: Too short")
            continue
        frame_count = get_frame_count(time_bin)
        pad_start = (frame_count * 300 - wave.shape[0]) // 2
        pad_end = frame_count * 300 - wave.shape[0] - pad_start
        wave = numpy.concatenate(
            [numpy.zeros([pad_start]), wave, numpy.zeros([pad_end])], axis=0
        )

        wave_16k = librosa.resample(
            wave, orig_sr=24000, target_sr=16000, res_type="kaiser_best"
        )
        pitch_rmvpe = (
            torch.from_numpy(rmvpe.infer_from_audio(wave_16k)).float().unsqueeze(0)
        )  # (1, frames)
        pitch = torch.nn.functional.interpolate(
            pitch_rmvpe.unsqueeze(1),  # (1, 1, frames)
            size=frame_count,
            mode="linear",
            align_corners=True,
        ).squeeze(
            1
        )  # (1, frames)
        if torch.any(torch.isnan(pitch)):
            pitch[torch.isnan(pitch)] = zero_value
        # pitch = pitch[:, :-1]
        result[name] = pitch

        print(".", end=" ")
        if count % 100 == 0:
            print(f"P{process_id} {count}/{len(lines)}")
    return result


def get_frame_count(i):
    return i * 20 + 20 + 40


def get_time_bin(sample_count):
    result = -1
    frames = sample_count // 300
    if frames >= 20:
        result = (frames - 20) // 20
    return result


if __name__ == "__main__":
    main()
