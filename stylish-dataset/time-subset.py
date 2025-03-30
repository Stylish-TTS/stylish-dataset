# Accept a train or val list on standard input and split it into many
# lists based on the duration of the clips.
#
# --wav is the directory of the wav files to check
# --out is the output directory where a bunch of segment lists will be spawned

import soundfile as sf
import argparse, pathlib, re, sys

parser = argparse.ArgumentParser()
parser.add_argument("--wav", default="wav/")
args = parser.parse_args()

wavdir = pathlib.Path(args.wav)

already = {}
time_bins = {}

time = 0.0

for line in sys.stdin:
    fields = line.strip().split("|")
    if fields[0].strip() in already:
        sys.stderr.write("DUPLICATE " + fields[0].strip())
    else:
        already[fields[0].strip()] = True
    audio, sample_rate = sf.read(str(wavdir / fields[0].strip()))
    time += audio.shape[0] / (300*80)
    if time < 25*60*60:
        print(line.strip())
    else:
        break

#print(time)
