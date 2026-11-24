"""AV1 video encoding shared by policy rollout and expert recovery."""

import os

import av


_DENSE_VIDEO_OPTIONS = {"g": "2", "crf": "30"}


class SuppressNativeStderr:
    def __enter__(self):
        self.saved_stderr = os.dup(2)
        self.devnull = open(os.devnull, "w")
        os.dup2(self.devnull.fileno(), 2)
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self.saved_stderr, 2)
        os.close(self.saved_stderr)
        self.devnull.close()


def add_dense_video_stream(container, width, height, fps):
    stream = container.add_stream(
        "libsvtav1",
        rate=fps,
        options=_DENSE_VIDEO_OPTIONS,
    )
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    return stream


def write_rollout_frame(container, stream, video_path, frame, policy_infer_freq):
    with SuppressNativeStderr():
        if container is None:
            h, w = frame.shape[:2]
            container = av.open(str(video_path), mode="w")
            stream = add_dense_video_stream(container, w, h, policy_infer_freq)

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    return container, stream
