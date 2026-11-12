"""Run the public Klygo API demo used in chapter 6, with saved verification evidence."""
import json
from run_pilot import ARTIFACTS, DATA, load_detector
from klygo import files, media, postprocessing as post, preprocessing as p, visual


if __name__ == "__main__":
    destination = ARTIFACTS / "demo"
    files.mkdir(destination, parents=True, exist_ok=True)
    source = DATA / "images/000000125778.jpg"
    image = media.LazyImage(source)
    detector = load_detector("yolo", "cuda:0")
    try:
        result = detector.predict(image, threshold=0.25, verbose=False)
        cleaned = post.compose(post.clip(), post.threshold(0.25))(result)
        frame = cleaned[0]
        annotated = visual.draw_bboxes(image.to_pil(), frame.boxes, labels=frame.labels,
                                       scores=frame.scores, font_size=14)
        media.save(destination / "prediction.png", annotated, overwrite=True)
        transformed = p.compose(p.letterbox((640, 640)))(media.LazyImage(source))
        transformed_result = detector.predict(transformed, verbose=False)[0]
        restored = transformed_result.restore(target="original")
        assert restored.image_size == image.size
        original_box = [[10, 20, 110, 120]]
        trace = p.get_trace(transformed)
        assert trace and trace[-1]["name"] == "letterbox"
        scale, padding = trace[-1]["scale"], trace[-1]["padding"]
        x1, y1, x2, y2 = original_box[0]
        moved = [[scale * x1 + padding[0], scale * y1 + padding[1],
                  scale * x2 + padding[0], scale * y2 + padding[1]]]
        recovered = p.restore_boxes(moved, trace)
        assert max(abs(a - b) for a, b in zip(original_box[0], recovered[0])) < 1e-6
        # Folder streaming is a finite smoke test; no claim about long-video memory is made.
        stream = detector.predict(str(DATA / "images"), stream=True, max_frames=6, batch=2, verbose=False)
        frames = list(stream)
        assert len(frames) == 6
        evidence = {"source": str(source), "original_size": image.size,
            "prediction_box_count": len(frame.objects), "labels": frame.labels,
            "transformed_size": transformed_result.image_size, "restored_size": restored.image_size,
            "trace": trace, "coordinate_round_trip_passed": True,
            "stream_frames": len(frames), "stream_frame_indices": [f.frame_index for f in frames]}
        files.save(destination / "verification.json", evidence, overwrite=True, verbose=False)
        print(json.dumps(evidence, indent=2))
    finally:
        detector.unload()
