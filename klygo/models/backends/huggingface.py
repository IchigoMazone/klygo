import torch
from typing import Any
from ..registry import registry
from ..exceptions import ModelLoadError, BackendExecutionError

@registry.register_backend("huggingface")
class HuggingFaceBackend:
    def __init__(self, model, processor=None, tokenizer=None, device: str = "cpu", **kwargs):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, model_path: str, device: str = "cpu", **kwargs):
        """Loads model weight and config from HuggingFace Hub or local directory using lazy imports."""
        try:
            import os
            from transformers import AutoProcessor, AutoTokenizer, AutoModelForZeroShotObjectDetection, AutoModel, VisionEncoderDecoderModel
            
            # Detect if model_path is a local path or directory to run in offline mode
            is_local = os.path.isdir(model_path) or os.path.exists(model_path)
            local_files_only = kwargs.get("local_files_only", is_local)
            
            hf_kwargs = {"local_files_only": local_files_only}
            if "trust_remote_code" in kwargs:
                hf_kwargs["trust_remote_code"] = kwargs["trust_remote_code"]
                
            processor = None
            tokenizer = None
            
            # Load appropriate model class
            if "grounding-dino" in model_path.lower():
                processor = AutoProcessor.from_pretrained(model_path, **hf_kwargs)
                model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, **hf_kwargs)
            elif "trocr" in model_path.lower():
                processor = AutoProcessor.from_pretrained(model_path, **hf_kwargs)
                model = VisionEncoderDecoderModel.from_pretrained(model_path, **hf_kwargs)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)
                except Exception:
                    pass
                try:
                    processor = AutoProcessor.from_pretrained(model_path, **hf_kwargs)
                except Exception:
                    pass
                
                # Detect causal LLM task/models (Qwen, Llama, Gemma, Phi, etc.)
                is_llm = kwargs.get("task") == "llm" or any(
                    name in model_path.lower() for name in ["qwen", "llama", "phi", "gpt", "mistral", "gemma"]
                )
                if is_llm and tokenizer is not None:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(model_path, **hf_kwargs)
                else:
                    model = AutoModel.from_pretrained(model_path, **hf_kwargs)
                
            return cls(model, processor=processor, tokenizer=tokenizer, device=device, **kwargs)
        except Exception as e:
            raise ModelLoadError(f"Failed to load HuggingFace model from {model_path}: {e}")

    @property
    def native(self):
        return self.model

    def predict(self, images: list, **kwargs) -> list:
        """Executes forward pass for vision tasks."""
        # 1. OCR (TrOCR)
        if "trocr" in self.model.__class__.__name__.lower():
            all_results = []
            for img in images:
                pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                all_results.append({"text": generated_text, "regions": []})
            return all_results
            
        # 2. Object Detection (Grounding DINO)
        elif hasattr(self.processor, "post_process_grounded_object_detection"):
            text_prompt = kwargs.get("text_prompt", "")
            if not text_prompt.strip().endswith("."):
                text_prompt = text_prompt.strip() + " ."
                
            all_results = []
            for img in images:
                inputs = self.processor(images=img, text=text_prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                width, height = img.size
                target_sizes = torch.tensor([[height, width]]).to(self.device)
                
                raw_results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    box_threshold=kwargs.get("box_threshold", 0.3),
                    text_threshold=kwargs.get("text_threshold", 0.25),
                    target_sizes=target_sizes
                )[0]
                
                all_results.append({
                    "boxes": raw_results["boxes"].cpu(),
                    "scores": raw_results["scores"].cpu(),
                    "labels": raw_results["labels"]
                })
            return all_results

        # 3. Image Classification
        else:
            from transformers import pipeline
            pipe = pipeline("image-classification", model=self.model, image_processor=self.processor, device=self.device)
            return [pipe(img) for img in images]

    def generate_text(self, prompt: str, **kwargs) -> dict:
        """Executes text generation for LLMs."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        temperature = kwargs.get("temperature", 0.7)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 128),
                temperature=temperature,
                do_sample=temperature > 0
            )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text, "tokens": len(outputs[0])}

    def generate_chat(self, messages: list, **kwargs) -> dict:
        """Executes chat template generation."""
        # Convert list of dicts messages to prompt using tokenizer's template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.generate_text(prompt, **kwargs)

    def generate_stream(self, prompt: str, **kwargs):
        """Streams text generation tokens."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        temperature = kwargs.get("temperature", 0.7)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=kwargs.get("max_tokens", 128),
            temperature=temperature,
            do_sample=temperature > 0
        )
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        return streamer

    def embed_text(self, text: str, **kwargs) -> list:
        """Generates embedding vector for text input."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].cpu().numpy().tolist()

    def embed_image(self, image, **kwargs) -> list:
        """Generates embedding vector for image input."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs) if hasattr(self.model, "get_image_features") else self.model(**inputs).pooler_output
        return outputs[0].cpu().numpy().tolist()

    def transcribe(self, audio_path: str, **kwargs) -> dict:
        """Transcribes audio file to text using pipeline."""
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition", 
            model=self.model, 
            feature_extractor=self.processor, 
            tokenizer=self.tokenizer, 
            device=self.device
        )
        out = pipe(audio_path)
        return {"text": out["text"], "chunks": out.get("chunks", [])}

    def train(self, dataset, args, **kwargs) -> Any:
        """Runs the native Hugging Face Trainer."""
        from transformers import Trainer
        eval_dataset = kwargs.get("eval_dataset", None)
        data_collator = kwargs.get("data_collator", None)
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            processing_class=self.tokenizer or self.processor
        )
        return trainer.train()

    def evaluate(self, dataset, args, **kwargs) -> dict:
        from transformers import Trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=dataset,
            processing_class=self.tokenizer or self.processor
        )
        return trainer.evaluate()

    def save(self, path: str):
        try:
            self.model.save_pretrained(path)
            if self.processor:
                self.processor.save_pretrained(path)
            if self.tokenizer:
                self.tokenizer.save_pretrained(path)
        except Exception as e:
            raise BackendExecutionError(f"Failed to save HF model to {path}: {e}")
