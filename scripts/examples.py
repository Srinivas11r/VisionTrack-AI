"""
Example usage of the detection system
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from backend.core.processor import VideoProcessor


def example_1_basic_detection():
    """Example 1: Basic object detection on video"""
    print("=== Example 1: Basic Detection ===\n")

    processor = VideoProcessor()

    # Process video with default settings (detect all objects)
    video_path = "test_video.mp4"  # Replace with your video
    output_path = "outputs/example1_output.mp4"

    if os.path.exists(video_path):
        logs = processor.process_video(video_path, output_path)
        print(f"\nDetected {len(logs)} unique objects")
    else:
        print(f"Video not found: {video_path}")


def example_2_filter_by_type():
    """Example 2: Detect only specific object types"""
    print("\n=== Example 2: Filter by Type ===\n")

    processor = VideoProcessor()

    # Only track people
    processor.set_specifications(
        {"object_type": "person", "color": "all", "size": "all", "confidence": 0.5}
    )

    video_path = "test_video.mp4"
    output_path = "outputs/example2_people_only.mp4"

    if os.path.exists(video_path):
        logs = processor.process_video(video_path, output_path)
        print(f"\nTracked {len(logs)} people")


def example_3_filter_by_color():
    """Example 3: Detect objects with specific color"""
    print("\n=== Example 3: Filter by Color ===\n")

    processor = VideoProcessor()

    # Track all red objects
    processor.set_specifications(
        {"object_type": "all", "color": "red", "size": "all", "confidence": 0.5}
    )

    video_path = "test_video.mp4"
    output_path = "outputs/example3_red_objects.mp4"

    if os.path.exists(video_path):
        logs = processor.process_video(video_path, output_path)
        print(f"\nTracked {len(logs)} red objects")


def example_4_specific_combination():
    """Example 4: Combine multiple filters"""
    print("\n=== Example 4: Specific Combination ===\n")

    processor = VideoProcessor()

    # Track large black cars only
    processor.set_specifications(
        {"object_type": "car", "color": "black", "size": "large", "confidence": 0.6}
    )

    video_path = "test_video.mp4"
    output_path = "outputs/example4_black_large_cars.mp4"

    if os.path.exists(video_path):
        logs = processor.process_video(video_path, output_path)

        print(f"\nTracked {len(logs)} black large cars")

        # Print detailed logs
        for log in logs:
            print(
                f"ID {log['id']}: appeared in {log['frame_count']} frames, "
                f"duration: {log['duration']:.2f}s"
            )


def example_5_webcam():
    """Example 5: Real-time webcam tracking"""
    print("\n=== Example 5: Webcam Tracking ===\n")

    processor = VideoProcessor()

    # Track people in real-time
    processor.set_specifications(
        {"object_type": "person", "color": "all", "size": "all", "confidence": 0.5}
    )

    print("Starting webcam... Press 'q' to quit")
    output_path = "outputs/example5_webcam.mp4"

    logs = processor.process_webcam(output_path)
    print(f"\nTracked {len(logs)} people during session")


def example_6_batch_processing():
    """Example 6: Process multiple videos"""
    print("\n=== Example 6: Batch Processing ===\n")

    processor = VideoProcessor()

    # Track backpacks in multiple videos
    processor.set_specifications(
        {"object_type": "backpack", "color": "all", "size": "all", "confidence": 0.5}
    )

    video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]

    total_tracks = 0

    for video_file in video_files:
        if os.path.exists(video_file):
            output_path = f"outputs/processed_{video_file}"
            print(f"\nProcessing {video_file}...")

            logs = processor.process_video(video_file, output_path)
            total_tracks += len(logs)

            print(f"Found {len(logs)} backpacks")
            processor.reset()  # Reset between videos

    print(f"\nTotal backpacks across all videos: {total_tracks}")


def example_7_custom_confidence():
    """Example 7: Adjust confidence threshold"""
    print("\n=== Example 7: Custom Confidence ===\n")

    processor = VideoProcessor()

    # High confidence only (reduce false positives)
    processor.set_specifications(
        {"object_type": "all", "color": "all", "size": "all", "confidence": 0.8}  # High confidence
    )

    video_path = "test_video.mp4"
    output_path = "outputs/example7_high_confidence.mp4"

    if os.path.exists(video_path):
        logs = processor.process_video(video_path, output_path)
        print(f"\nHigh-confidence detections: {len(logs)}")


def main():
    print("Object Detection & Tracking - Usage Examples\n")
    print("Choose an example to run:")
    print("1. Basic detection (all objects)")
    print("2. Filter by type (people only)")
    print("3. Filter by color (red objects)")
    print("4. Specific combination (black large cars)")
    print("5. Webcam tracking")
    print("6. Batch processing")
    print("7. Custom confidence threshold")
    print("0. Run all examples")

    choice = input("\nEnter choice (0-7): ").strip()

    examples = {
        "1": example_1_basic_detection,
        "2": example_2_filter_by_type,
        "3": example_3_filter_by_color,
        "4": example_4_specific_combination,
        "5": example_5_webcam,
        "6": example_6_batch_processing,
        "7": example_7_custom_confidence,
    }

    if choice == "0":
        for func in examples.values():
            func()
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
