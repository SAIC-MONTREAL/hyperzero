import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env, text=None):
        if self.enabled:
            if hasattr(env, 'physics'):
                # DM control
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            elif hasattr(env, 'env'):
                # Meta World
                frame = env.env.render(offscreen=True, resolution=(self.render_size, self.render_size))
            else:
                # OpenAI Gym
                frame = env.render()
            if text is not None:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype("SFCompact.ttf", 24)
                draw.text((50, 50), text, (255, 255, 255), font)
                frame = np.array(img)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
