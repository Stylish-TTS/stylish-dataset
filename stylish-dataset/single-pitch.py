import pathlib, sys

import click
import librosa
import numpy
import soundfile
import torch
import torchaudio
import torchcrepe
import pyworld

from safetensors.torch import save_file

device = "cuda"


@click.command()
@click.option("--wavdir", default="wav", type=str)
@click.option("--inpath", default="list.txt", type=str)
@click.option("--outpath", default="pitch.safetensors", type=str)
def main(wavdir, inpath, outpath):
    wavdir = pathlib.Path(wavdir)
    result = calculate_pitch(pathlib.Path(inpath), wavdir)
    save_file(result, outpath)


def calculate_pitch(path, wavdir):
    result = {}
    count = 0
    with path.open("r") as f:
        for line in f:
            fields = line.split("|")
            name = fields[0]
            wave, sr = soundfile.read(wavdir / name)
            if sr != 24000:
                sys.stderr.write(f"Skipping {name}: Wrong sample rate ({sr})")
            if wave.shape[-1] == 2:
                wave = wave[:, 0].squeeze()
            time_bin = get_time_bin(wave.shape[0])
            if time_bin == -1:
                sys.stderr.write(f"Skipping {name}: Too short\n")
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
                sys.stderr.write("D")
                f0, t = pyworld.dio(wave, 24000, frame_period=frame_period)
            pitch = pyworld.stonemask(wave, f0, t, 24000)
            pitch = torch.from_numpy(pitch).float().unsqueeze(0)
            if torch.any(torch.isnan(pitch)):
                pitch[torch.isnan(pitch)] = self.zero_value

            # wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            #
            # with torch.no_grad():
            #    wave = torch.from_numpy(wave).float().unsqueeze(0)
            #    fmin = 0
            #    fmax = 8000
            #    model = "full"
            #    pitch, periodicity = torchcrepe.predict(
            #        wave,
            #        16000,
            #        200,
            #        fmin,
            #        fmax,
            #        model,
            #        batch_size=8192,
            #        device=device,
            #        return_periodicity=True,
            #    )
            #    periodicity = torchcrepe.threshold.Silence(-60.0)(
            #        periodicity, wave, 16000, 200
            #    )
            #    pitch = torchcrepe.threshold.At(0.21)(pitch, periodicity)
            pitch = pitch[:, :-1]
            result[name] = pitch
            count += 1
            sys.stderr.write(".")
            if count % 50 == 0:
                sys.stderr.write(str(count) + "\n")
            sys.stderr.flush()
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
