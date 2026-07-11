from __future__ import annotations
from PIL import Image
from .base import VLM
from .messages import Message, GenParams, Response


class Qwen3VLLocal(VLM):
    """Qwen3-VL via transformers, lazy-loaded."""

    def __init__(self, model: str = "Qwen/Qwen3-VL-7B-Instruct",
                 device: str | None = None):
        self.model_name = model
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16,
            device_map=self.device or "auto",
        )

    def _to_qwen(self, messages: list[Message]):
        conv = []
        images_flat: list[Image.Image] = []
        for m in messages:
            content = []
            for im in m.images:
                content.append({"type": "image", "image": im})
                images_flat.append(im)
            if m.text:
                content.append({"type": "text", "text": m.text})
            conv.append({"role": m.role, "content": content})
        return conv, images_flat

    def generate(self, messages, params):
        self._load()
        conv, images = self._to_qwen(messages)
        text = self._processor.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self._model.device)
        gen_kwargs = dict(
            max_new_tokens=params.max_new_tokens,
            do_sample=params.temperature > 0.0,
            temperature=max(params.temperature, 1e-5),
            top_p=params.top_p,
        )
        out = self._model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        text_out = self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        for s in params.stop:
            i = text_out.find(s)
            if i != -1:
                text_out = text_out[:i]
        return Response(
            text=text_out.strip(),
            usage={"input": int(inputs["input_ids"].shape[1]),
                   "output": int(trimmed.shape[1])},
        )


class Qwen3VLEndpoint(VLM):
    """OpenAI-compatible endpoint (vLLM). Supports system + multi-turn + images."""

    def __init__(self, endpoint: str, model: str = "Qwen/Qwen3-VL-7B-Instruct",
                 max_retries: int = 3, api_key: str = "EMPTY"):
        self.endpoint = endpoint
        self.model = model
        self.max_retries = max_retries
        self.api_key = api_key
        self._client = None

    def _load(self):
        if self._client is not None:
            return
        from openai import OpenAI
        self._client = OpenAI(base_url=self.endpoint, api_key=self.api_key)

    def _to_openai(self, messages):
        import base64, io
        out = []
        for m in messages:
            parts = []
            for im in m.images:
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:image/png;base64,{b64}"}})
            if m.text:
                parts.append({"type": "text", "text": m.text})
            out.append({"role": m.role, "content": parts if m.images else m.text})
        return out

    def generate(self, messages, params):
        self._load()
        payload = self._to_openai(messages)
        last_err = None
        for _ in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=payload,
                    max_tokens=params.max_new_tokens,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    stop=params.stop or None,
                )
                text = resp.choices[0].message.content or ""
                usage = {}
                if resp.usage:
                    usage = {"input": getattr(resp.usage, "prompt_tokens", 0),
                             "output": getattr(resp.usage, "completion_tokens", 0)}
                return Response(text=text.strip(), usage=usage, raw=resp)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"VLM endpoint failed after retries: {last_err}")


def build_vlm(cfg) -> VLM:
    b = cfg.vlm.backend
    if b == "qwen3vl_local":
        return Qwen3VLLocal(model=cfg.vlm.model)
    if b == "qwen3vl_endpoint":
        if not cfg.vlm.endpoint:
            raise ValueError("qwen3vl_endpoint requires cfg.vlm.endpoint")
        return Qwen3VLEndpoint(endpoint=cfg.vlm.endpoint, model=cfg.vlm.model,
                               max_retries=cfg.vlm.max_retries)
    raise ValueError(f"Unknown VLM backend: {b}")
