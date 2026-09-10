import os
import cv2
import time
import tracemalloc
import tempfile
import numpy as np
import klygo as kg
import gc

def create_dummy_video(filename, num_frames=300, width=640, height=480, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    print(f"Created {filename} with {num_frames} frames.")

class MockDetector:
    def predict(self, source, stream=False, **kwargs):
        frames = kg.media.load(source, stream=stream)
        def generator():
            for i, frame in enumerate(frames):
                det = kg.outputs.detect.Detection(
                    source_image=frame,
                    objects=[kg.outputs.detect.Box(id=0, label="mock", score=0.9, box=[10, 10, 100, 100], parent_image=frame, pad=0)],
                )
                yield det
                
        if stream:
            return kg.outputs.detect.Detections(generator(), source_type="video", fps=frames.fps, stream=True, total_frames=len(frames))
        else:
            return kg.outputs.detect.Detections(list(generator()), source_type="video", fps=frames.fps, stream=False)

def test_deep_e2e_pipeline():
    print("\n--- Deep E2E Pipeline Test: Load -> Predict -> Slice -> Save ---")
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, 'e2e_test_video.mp4')
    output_path = os.path.join(temp_dir, 'e2e_test_out.mp4')
    
    try:
        create_dummy_video(video_path, num_frames=300)
        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()
        
        detector = MockDetector()
        print("Running prediction (stream=True)...")
        results = detector.predict(video_path, stream=True)
        
        print("Slicing detections (results[50:150])...")
        sliced_results = results[50:150]
        
        assert len(sliced_results) == 100, f"Expected 100 sliced detections, got {len(sliced_results)}"
        assert sliced_results[0].source_image.frame_index == 50, "Frame index metadata lost"
        
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats)
        mb_diff = total_diff / 1024 / 1024
        print(f"Memory overhead before saving: {mb_diff:.2f} MB")
        assert mb_diff < 10, "Memory leaked significantly"
        
        print(f"Saving sliced results to {output_path}...")
        start_time = time.time()
        sliced_results.save(output_path, fps=30)
        save_time = time.time() - start_time
        print(f"Saving video took {save_time:.2f}s")
        
        assert os.path.exists(output_path), "Output video was not created!"
        
        cap = cv2.VideoCapture(output_path)
        out_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert out_frames == 100, f"Expected 100 frames, got {out_frames}"
        
        print("[OK] E2E Pipeline fully functional and Zero-RAM compliant!")
        
    finally:
        if 'results' in locals() and hasattr(results, 'close'):
            results.close()
        gc.collect()
        time.sleep(0.5)
        for p in [video_path, output_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

if __name__ == "__main__":
    test_deep_e2e_pipeline()
