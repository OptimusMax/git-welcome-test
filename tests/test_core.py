import shutil
import subprocess

import numpy as np
import pytest

from dubber import media
from dubber.segments import Segment, merge_short_segments, read_srt, write_srt
from dubber.timeline import assemble, plan_timeline
from dubber.translate import translate_segments


def test_srt_roundtrip(tmp_path):
    segs = [Segment(0.5, 2.25, "Hello", "Привіт"), Segment(3661.001, 3662.5, "World", "Світ")]
    p = tmp_path / "a.srt"
    write_srt(segs, p)
    assert "01:01:01,001 --> 01:01:02,500" in p.read_text(encoding="utf-8")
    back = read_srt(p)
    assert [s.text for s in back] == ["Привіт", "Світ"]
    assert back[0].start == pytest.approx(0.5) and back[1].end == pytest.approx(3662.5)


def test_merge_short_segments():
    segs = [Segment(0, 0.4, "Hi"), Segment(0.5, 3.0, "there friend"), Segment(5, 8, "Next")]
    merged = merge_short_segments(segs)
    assert [s.text for s in merged] == ["Hi there friend", "Next"]
    assert merged[0].start == 0 and merged[0].end == 3.0


def test_plan_fits_without_speedup():
    segs = [Segment(0, 2, "a"), Segment(3, 5, "b")]
    pl = plan_timeline(segs, [1.5, 1.0], total_duration=6)
    assert [p.tempo for p in pl] == [1.0, 1.0]
    assert [p.start for p in pl] == [0, 3]


def test_plan_speeds_up_and_shifts():
    segs = [Segment(0, 2, "a"), Segment(2, 4, "b")]
    pl = plan_timeline(segs, [4.0, 1.0], total_duration=10, max_speed=1.5)
    assert pl[0].tempo == pytest.approx(1.5)
    # перша фраза не вміщається навіть із прискоренням — друга зсувається без накладання
    assert pl[1].start >= pl[0].start + pl[0].duration


def test_assemble_places_clips():
    sr = 1000
    clips = [np.ones(500, dtype=np.float32) * 0.5, None]
    from dubber.timeline import Placement

    track = assemble(clips, [Placement(1.0, 1.0, 0.5), Placement(2.0, 1.0, 0)], sr, 3.0, fade_ms=0)
    assert len(track) == 3000
    assert track[999] == 0 and track[1000] == pytest.approx(0.5) and track[1500] == 0


def test_translate_segments_batches():
    segs = [Segment(0, 1, f"t{i}") for i in range(5)]
    out = translate_segments(segs, lambda texts: [t.upper() for t in texts])
    assert [s.translation for s in out] == ["T0", "T1", "T2", "T3", "T4"]


def test_atempo_chain():
    assert media.atempo_chain(1.0) == ""
    assert media.atempo_chain(3.0) == "atempo=2.0,atempo=1.5000"
    assert media.atempo_chain(0.25) == "atempo=0.5,atempo=0.5000"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="потрібен ffmpeg")
def test_mux_end_to_end(tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-c:a", "aac",
         "-shortest", str(video)],
        check=True,
    )
    sr = media.SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    seg = Segment(0.5, 1.5, "x", "ікс")
    dub = media.save_wav(assemble([tone], plan_timeline([seg], [1.0], 3.0), sr, 3.0), tmp_path / "d.wav")
    srt = tmp_path / "s.srt"
    write_srt([seg], srt)
    out = media.mux(video, dub, tmp_path / "out.mp4", original_volume=0.1, keep_original_track=True, subtitles=srt)
    assert media.probe_duration(out) == pytest.approx(3.0, abs=0.2)
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert streams.count("audio") == 2 and "video" in streams and "subtitle" in streams
    # load_audio з прискоренням коротшає вдвічі
    assert len(media.load_audio(dub, sr, tempo=2.0)) / sr == pytest.approx(1.5, abs=0.1)
