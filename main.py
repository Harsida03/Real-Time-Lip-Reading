import torch
import hydra
import cv2
import time
from pipelines.pipeline import InferencePipeline
import numpy as np
from datetime import datetime
from ollama import chat
from pydantic import BaseModel
import keyboard
from concurrent.futures import ThreadPoolExecutor
import os
import io
import contextlib
import sys


# Pydantic model for structured LLM output
class LipNetOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class LipNet:
    def __init__(self):
        self.vsr_model = None
        self.recording = False
        self.executor = ThreadPoolExecutor(max_workers=1)

        # video parameters
        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

    def perform_inference(self, video_path):
        """Perform inference and print the results in terminal (without AVSR token/label prints)."""
        print(f"\n[INFO] Running inference on: {video_path}")

        # Suppress stdout and stderr from the inference call (removes token/label printouts)
        # Warning: this will hide all prints/warnings coming from inside the model code.
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            output = self.vsr_model(video_path)

        # Print the clean outputs we want
        print(f"[RAW TRANSCRIPTION]: {output}")

        # send transcription to LLM for correction
        response = chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are an assistant that helps correct the output of a lipreading model. "
                        "The input text is a raw transcription from a video-to-text system, so it may contain errors. "
                        "Fix mistranscribed words, ensure proper grammar and punctuation, but do NOT add or remove content. "
                        "Return both a list of changes and the corrected text. "
                        "Response format: {'list_of_changes': str, 'corrected_text': str}."
                    )
                },
                {'role': 'user', 'content': f"Transcription:\n\n{output}"}
            ],
            format=LipNetOutput.model_json_schema()
        )

        # parse LLM response
        chat_output = LipNetOutput.model_validate_json(response.message.content)

        # ensure ending punctuation
        if chat_output.corrected_text and chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        # print both raw and corrected text
        print("\n[LLM CORRECTIONS]")
        print(f"List of Changes: {chat_output.list_of_changes}")
        print(f"Corrected Text: {chat_output.corrected_text}\n")
        print("-" * 80)

        return {
            "output": chat_output.corrected_text,
            "video_path": video_path
        }

    def start_webcam(self):
        """Start webcam, record short clips, and run inference in background."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0

        print("\n[INFO] Press 'Alt' to start/stop recording. Press 'q' to quit.\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quitting and cleaning up temporary files...")
                for file in os.listdir():
                    if file.startswith(self.output_prefix) and file.endswith('.mp4'):
                        try:
                            os.remove(file)
                        except Exception:
                            pass
                break

            current_time = time.time()

            if current_time - last_frame_time >= self.frame_interval:
                ret, frame = cap.read()
                if ret:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.frame_compression]
                    _, buffer = cv2.imencode('.jpg', frame, encode_param)
                    compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)

                    if self.recording:
                        if out is None:
                            output_path = f"{self.output_prefix}{time.time_ns() // 1_000_000}.mp4"
                            out = cv2.VideoWriter(
                                output_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                self.fps,
                                (frame_width, frame_height),
                                False
                            )

                        out.write(compressed_frame)
                        last_frame_time = current_time
                        # Recording indicator on preview (not saved)
                        cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                        frame_count += 1

                    elif not self.recording and frame_count > 0:
                        if out is not None:
                            out.release()

                        # only run inference if the video is at least 2 seconds long
                        if frame_count >= self.fps * 2:
                            print(f"[INFO] Processing clip ({frame_count/self.fps:.1f}s)...")
                            futures.append(self.executor.submit(self.perform_inference, output_path))
                        else:
                            # short clip — delete file
                            try:
                                os.remove(output_path)
                            except Exception:
                                pass

                        output_path = ""
                        frame_count = 0
                        out = None

                    cv2.imshow('LipNet', cv2.flip(compressed_frame, 1))

            # handle finished futures in order
            for fut in futures[:]:
                if fut.done():
                    try:
                        result = fut.result()
                        if os.path.exists(result["video_path"]):
                            try:
                                os.remove(result["video_path"])
                            except Exception:
                                pass
                    except Exception:
                        # silently ignore inference task errors
                        pass
                    finally:
                        futures.remove(fut)
                else:
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def on_action(self, event):
        if event.event_type == keyboard.KEY_DOWN and event.name == 'alt':
            self.recording = not self.recording
            state = "STARTED" if self.recording else "STOPPED"
            print(f"[INFO] Recording {state}.")


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    lipnet = LipNet()
    keyboard.hook(lambda e: lipnet.on_action(e))

    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
    lipnet.vsr_model = InferencePipeline(cfg.config_filename, device=device, detector=cfg.detector, face_track=True)

    print("[INFO] Model loaded successfully!")
    lipnet.start_webcam()


if __name__ == "__main__":
    main()
