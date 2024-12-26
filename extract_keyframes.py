from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import logging as transformers_logging
import warnings
import cv2
import numpy as np
import torch.nn.functional as F
import logging
import re

# Suppress warnings from the transformers library
transformers_logging.set_verbosity_error()

# Optional: suppress warnings globally
warnings.filterwarnings("ignore")


class LectureKeyframeExtractor:
    """Extract keyframes from lecture videos with slide transitions and generate descriptions."""

    def __init__(self, video_folder: str, output_dir: str, model, processor, threshold: float = 0.85):
        """
        Initialize the keyframe extractor.

        Args:
            video_folder: Path to the folder containing video files
            output_dir: Directory to save extracted keyframes
            model: Llava Model
            processor: LlavaNextProcessor instance for generating descriptions
            threshold: SSIM threshold for detecting frame changes (default: 0.85)
        """
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        self.video_folder = video_folder
        self.output_dir = output_dir
        self.model = model
        self.processor = processor
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        log_file = os.path.join(output_dir, "keyframe_extraction.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Keep console logging
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.data = []

        """
        Initialize the keyframe extractor.

        Args:
            video_folder: Path to the folder containing video files
            output_dir: Directory to save extracted keyframes
            threshold: SSIM threshold for detecting frame changes (default: 0.7)
        """
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        self.video_folder = video_folder
        self.output_dir = output_dir
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging
        log_file = os.path.join(output_dir, "keyframe_extraction.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Keep console logging
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        self.data = []

    def ssim_torch(self, img1, img2, window_size=11, size_average=True):
        """
        Calculate SSIM between two images using PyTorch.

        Args:
            img1: First image as a NumPy array
            img2: Second image as a NumPy array
            window_size: Size of the sliding window (default: 11)
            size_average: Whether to average the SSIM map (default: True)

        Returns:
            SSIM value.
        """
        img1 = torch.tensor(img1).float().unsqueeze(0).unsqueeze(0)
        img2 = torch.tensor(img2).float().unsqueeze(0).unsqueeze(0)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean().item()
        else:
            return ssim_map

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop the frame to focus on the presentation slide area.

        Args:
            frame: Input frame

        Returns:
            Cropped frame
        """
        height, width, _ = frame.shape
        x_end = int(width * 0.7)    # 70% of width
        y_start = int(height * 0.05) # 5% from top
        y_end = int(height * 0.95)   # 95% of height

        return frame[y_start:y_end, :x_end]

    def generate_description(self, image_path: str) -> str:
        """
        Generate a description for the given image.

        Args:
            image_path: Path to the image file

        Returns:
            A string description of the image.
        """
        try:
            # Load and preprocess the image
            image = Image.open(image_path)
            self.logger.info(f"Processing image {image_path}")

            # Define a conversation history to use for the prompt
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract key details from this lecture recording image, focusing on text from slides, code snippets, algorithms, diagrams, annotations, and any highlighted points. Additionally, analyze relationships between visual elements and infer any subtle messages or overarching themes conveyed. Summarize the content concisely while avoiding unnecessary details."},
                        {"type": "image"},
                    ],
                },
            ]

            # Generate prompt
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

            # Prepare inputs for the model
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            ).to(self.device)

            # Generate output
            output = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)

            # Decode the generated text
            description_full = self.processor.decode(output[0], skip_special_tokens=True)

            # Remove unnecessary instructions or tags from the description
            description_cleaned = re.sub(r"\[INST\].*?\[/INST\]", "", description_full, flags=re.DOTALL).strip()

            # Extract the text after "ASSISTANT:" to get only the response
            if "ASSISTANT:" in description_cleaned:
                description = description_cleaned.split("ASSISTANT:")[1].strip()
            else:
                description = description_cleaned

            return description

        except Exception as e:
            self.logger.error(f"Error generating description for {image_path}: {e}")
            return "Description generation failed"

    def extract_keyframes(self, sample_rate: int = 2) -> None:
        """
        Extract keyframes from all videos in the folder and generate descriptions immediately.

        Args:
            sample_rate: Number of seconds between frame sampling (default: 2)
        """
        video_files = [f for f in os.listdir(self.video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]

        if not video_files:
            self.logger.error(f"No video files found in folder: {self.video_folder}")
            return

        for video_file in video_files:
            video_path = os.path.join(self.video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            self.logger.info(f"Processing video: {video_file}")

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video file: {video_file}")

                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_interval = int(fps * sample_rate)

                keyframe_index = 0
                previous_frame_gray = None
                first_keyframe_extracted = False

                with tqdm(total=total_frames, desc=f"Processing {video_file}", unit="frame") as pbar:
                    for frame_index in range(0, total_frames, frame_interval):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        success, frame = cap.read()

                        if not success:
                            break

                        # Crop the frame to focus only on slide area
                        cropped_frame = self.crop_frame(frame)

                        # Convert to grayscale after cropping
                        current_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

                        save_frame = False
                        if previous_frame_gray is None or not first_keyframe_extracted:
                            save_frame = True
                        else:
                            similarity = self.ssim_torch(previous_frame_gray, current_frame_gray)
                            save_frame = similarity < self.threshold

                        if save_frame:
                            keyframe_filename = os.path.join(self.output_dir, f"{video_name}_keyframe_{keyframe_index:04d}.jpg")

                            # Use JPEG quality of 95 for good balance of quality and size
                            cv2.imwrite(keyframe_filename, cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                            timestamp = frame_index / fps
                            description = self.generate_description(keyframe_filename)

                            # Append data to list
                            self.data.append({
                                "video_name": video_name,
                                "filename": os.path.basename(keyframe_filename),
                                "timestamp_seconds": timestamp,
                                "timestamp_formatted": f"{int(timestamp // 3600):02d}:{int((timestamp % 3600) // 60):02d}:{int(timestamp % 60):02d}",
                                "description": description
                            })

                            keyframe_index += 1
                            first_keyframe_extracted = True

                        previous_frame_gray = current_frame_gray
                        pbar.update(frame_interval)

                cap.release()
                
                excel_path = os.path.join(self.output_dir, "keyframes_info.xlsx")
                # Save metadata to Excel after processing each video
                if os.path.exists(excel_path):
                    # Load existing data and append the new data
                    existing_df = pd.read_excel(excel_path, engine="openpyxl")
                    new_df = pd.DataFrame(self.data)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    # No existing file, create new DataFrame
                    combined_df = pd.DataFrame(self.data)

                # Save the combined data to Excel
                combined_df.to_excel(excel_path, index=False, engine="openpyxl")
                self.logger.info(f"Updated metadata saved to {excel_path} for video: {video_name}")
                # Clear data for next video
                self.data.clear()

            except Exception as e:
                self.logger.error(f"Error processing video {video_file}: {str(e)}")
                continue

if __name__ == "__main__":
    video_folder = "Videos"
    output_dir = "keyframes_output"

    try:
        # Dynamically select device (CUDA if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load model to GPU
        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)  # Move model to GPU (device 0)

        # Load the processor
        processor = LlavaNextProcessor.from_pretrained(model_id, cache_dir="model")
        extractor = LectureKeyframeExtractor(video_folder, output_dir, model, processor, threshold=0.3)
        extractor.extract_keyframes(sample_rate=5)
    except Exception as e:
        logging.error(f"Failed to extract keyframes: {str(e)}")
