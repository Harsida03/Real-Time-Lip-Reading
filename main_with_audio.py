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
from audio_module import AudioCapture
from fusion import AudioVisualFusion


class LipNetOutput(BaseModel):
    list_of_changes: str
    corrected_text: str


class AudioVisualLipNet:
    def __init__(self):
        self.vsr_model = None
        self.recording = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        self.audio_capture = AudioCapture()
        self.fusion_module = AudioVisualFusion(audio_weight=0.6, visual_weight=0.4)
        self.audio_stream = None

        self.output_prefix = "webcam"
        self.res_factor = 3
        self.fps = 16
        self.frame_interval = 1 / self.fps
        self.frame_compression = 25

    def perform_inference(self, video_path, audio_text):
        print(f"\n[INFO] Running inference on: {video_path}")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            visual_output = self.vsr_model(video_path)

        print(f"[VISUAL PREDICTION]: {visual_output}")
        print(f"[AUDIO PREDICTION]: {audio_text if audio_text else 'None (no audio detected)'}")

        fused_text, fusion_method = self.fusion_module.fuse_predictions(
            visual_text=visual_output,
            audio_text=audio_text,
            visual_confidence=0.5
        )

        print(f"[FUSION METHOD]: {fusion_method}")
        print(f"[FUSED RESULT]: {fused_text}")

        response = chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'system',
                    'content': (
                        "You are an assistant that helps correct the output of a multimodal speech recognition system. "
                        "The input combines lip-reading and audio recognition. "
                        "Fix any errors, ensure proper grammar and punctuation, but do NOT add or remove content. "
                        "Return both a list of changes and the corrected text. "
                        "Response format: {'list_of_changes': str, 'corrected_text': str}."
                    )
                },
                {'role': 'user', 'content': f"Transcription:\n\n{fused_text}"}
            ],
            format=LipNetOutput.model_json_schema()
        )

        chat_output = LipNetOutput.model_validate_json(response.message.content)

        if chat_output.corrected_text and chat_output.corrected_text[-1] not in ['.', '?', '!']:
            chat_output.corrected_text += '.'

        print("\n[LLM CORRECTIONS]")
        print(f"List of Changes: {chat_output.list_of_changes}")
        print(f"Final Corrected Text: {chat_output.corrected_text}\n")
        print("-" * 80)

        return {
            "output": chat_output.corrected_text,
            "video_path": video_path
        }

    def start_webcam(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 // self.res_factor)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 // self.res_factor)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.audio_stream = self.audio_capture.get_audio_stream()
        self.audio_stream.start()

        last_frame_time = time.time()
        futures = []
        output_path = ""
        out = None
        frame_count = 0
        audio_text = None

        print("\n[INFO] Audio-Visual Lip Reading System Started!")
        print("[INFO] Press 'Alt' to start/stop recording. Press 'q' to quit.\n")

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
                            self.audio_capture.start_recording()

                        out.write(compressed_frame)
                        last_frame_time = current_time
                        cv2.circle(compressed_frame, (frame_width - 20, 20), 10, (0, 0, 0), -1)
                        frame_count += 1

                    elif not self.recording and frame_count > 0:
                        if out is not None:
                            out.release()
                        
                        audio_text = self.audio_capture.stop_recording()

                        if frame_count >= self.fps * 2:
                            print(f"[INFO] Processing clip ({frame_count/self.fps:.1f}s)...")
                            futures.append(
                                self.executor.submit(self.perform_inference, output_path, audio_text)
                            )
                        else:
                            try:
                                os.remove(output_path)
                            except Exception:
                                pass

                        output_path = ""
                        frame_count = 0
                        out = None
                        audio_text = None

                    cv2.imshow('Audio-Visual LipNet', cv2.flip(compressed_frame, 1))

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
                        pass
                    finally:
                        futures.remove(fut)
                else:
                    break

        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        
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
    lipnet = AudioVisualLipNet()
    keyboard.hook(lambda e: lipnet.on_action(e))

    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
    lipnet.vsr_model = InferencePipeline(cfg.config_filename, device=device, detector=cfg.detector, face_track=True)

    print("[INFO] Audio-Visual Model loaded successfully!")
    lipnet.start_webcam()


if __name__ == "__main__":
    main()
