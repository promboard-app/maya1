"""promboard.maya1 — Maya1(maya-research) 글로 목소리 설계 노드 코드.

노드 하나(maya1)가 글을 읽어 목소리 WAV(24kHz 모노)를 낸다. 목소리는 영어 설명 글(성별 · 나이 · 억양 · 음높이 · 음색 · 말투)로
정하고, 읽을 글 안에 <laugh> · <sigh> · <whisper> 같은 감정 태그를 넣을 수 있다(복제 없음).
흐름은 공식 카드 「Transformers inference example」 그대로다:
  프롬프트 = SOH + BOS + '<description="설명"> 글' + TEXT_EOT + EOH + SOA + SOS(토큰 번호는 카드 상수)
  → model.generate(temperature, top_p 0.9, repetition_penalty 1.1, max_new_tokens 2048, min_new_tokens 28, eos = 말 끝 토큰)
  → SNAC 토큰(128266~156937)만 모아 7개 = 한 프레임 → 세 층(1 · 2 · 4)으로 풀기 → SNAC 24kHz 디코더 → 앞 2048 샘플 버림.
긴 글: 공식 카드가 '알맞은 길이로 나눠 만들라'고 하므로 문장 경계에서 MAX_CHUNK_CHARS 안쪽으로 나눠 같은 설명으로 차례로 만들고 잇는다.
  조각 사이에 목소리가 조금 달라질 수 있다(복제가 없는 모델이라 같은 설명만 이어 쓴다).

모델(manifest requires.models, 허깅페이스에서 받는다)
  maya-research/maya1        Llama 구조 3.3B, bf16 6.7GB (Apache-2.0) — 표준 LlamaForCausalLM 이라 원격 코드 없이 읽는다
  hubertsiuzdak/snac_24khz   SNAC 24kHz 디코더 79MB (MIT)
노드 코드 약속(NODES · ctx · 입출력 모양)은 PromBoard docs/Scrum4/02_스키마-v0.md §6.
"""
import os
import re
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 받은 파일만 쓴다(실행 환경은 인터넷이 막혀 있다)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_FILE = "maya-research/maya1/config.json"
SNAC_FILE = "hubertsiuzdak/snac_24khz/config.json"
SAMPLE_RATE = 24000

# 공식 카드 예제 상수
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009
MAX_NEW_TOKENS = 2048  # 한 조각 약 25초(7토큰 = SNAC 한 프레임 ≈ 85ms)
MIN_NEW_TOKENS = 28  # SNAC 4프레임
WARMUP_SAMPLES = 2048  # 디코더 앞부분 버림(공식 예제)

MAX_CHUNK_CHARS = 280  # 한 조각에 넣을 글 — 25초 안에 읽히는 길이
MAX_TEXT_CHARS = 3000  # 한 번에 읽을 글 상한(약 3분 분량) — 앱이 노드 한 번에 주는 시간(만들기 15분)에 끝나게(소리 길이의 2배쯤 걸린다)

_STATE = {}


def _dtype(device):
    import torch

    return torch.float32 if device == "cpu" else torch.bfloat16


def _models(ctx):
    device = ctx.device
    if _STATE.get("device") == device:
        return _STATE["model"], _STATE["tokenizer"], _STATE["snac"]
    import torch
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ctx.progress(0.02, "load-model")  # 올리는 동안 파이썬이 오래 멈춘다 — 바로 앞에 알린다
    start = time.monotonic()
    _STATE.clear()
    path = os.path.dirname(ctx.model_path(MODEL_FILE))
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=_dtype(device), device_map=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path)
        snac = SNAC.from_pretrained(os.path.dirname(ctx.model_path(SNAC_FILE))).to(device).eval()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError(
            "그래픽카드 메모리가 모자라 모델을 올리지 못했습니다(약 8GB 필요). 다른 AI 작업 · 프로그램을 닫고 다시 해 주세요."
        ) from None
    _STATE.update(device=device, model=model, tokenizer=tokenizer, snac=snac)
    ctx.log(f"Maya1 + SNAC loaded on {device} in {time.monotonic() - start:.1f}s")
    return model, tokenizer, snac


def build_prompt(tokenizer, description, text):
    """공식 카드 build_prompt 그대로."""
    soh = tokenizer.decode([SOH_ID])
    eoh = tokenizer.decode([EOH_ID])
    soa = tokenizer.decode([SOA_ID])
    sos = tokenizer.decode([CODE_START_TOKEN_ID])
    eot = tokenizer.decode([TEXT_EOT_ID])
    return soh + tokenizer.bos_token + f'<description="{description}"> {text}' + eot + eoh + soa + sos


def unpack_snac_from_7(snac_tokens):
    """7토큰 프레임 → 세 층(공식 카드 unpack_snac_from_7 그대로)."""
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    l1, l2, l3 = [], [], []
    for i in range(frames):
        s = snac_tokens[i * 7:(i + 1) * 7]
        l1.append((s[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([(s[1] - CODE_TOKEN_OFFSET) % 4096, (s[4] - CODE_TOKEN_OFFSET) % 4096])
        l3.extend([(s[2] - CODE_TOKEN_OFFSET) % 4096, (s[3] - CODE_TOKEN_OFFSET) % 4096,
                   (s[5] - CODE_TOKEN_OFFSET) % 4096, (s[6] - CODE_TOKEN_OFFSET) % 4096])
    return [l1, l2, l3]


def split_text(text):
    """문장 끝(. ! ? 와 줄바꿈)에서 나눠 MAX_CHUNK_CHARS 안쪽 조각으로 묶는다. 한 문장이 너무 길면 쉼표 · 띄어쓰기에서 자른다."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    pieces = []
    for s in sentences:
        while len(s) > MAX_CHUNK_CHARS:
            cut = max(s.rfind(", ", 0, MAX_CHUNK_CHARS), s.rfind(" ", 0, MAX_CHUNK_CHARS))
            cut = cut if cut > MAX_CHUNK_CHARS // 2 else MAX_CHUNK_CHARS
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            pieces.append(s)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > MAX_CHUNK_CHARS:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def maya1(ctx, inputs, settings):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    text = (inputs.get("text") or "").strip()
    if not text:
        raise ValueError("읽을 글을 이어 주세요.")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"글이 너무 깁니다({len(text)}자). 한 번에 {MAX_TEXT_CHARS}자까지 읽습니다 — 나눠서 이어 주세요.")
    description = " ".join((settings["description"] or "").split())
    if not description:
        raise ValueError("설정 '목소리 설명'에 원하는 목소리를 영어로 적어 주세요.")
    description = description.replace('"', "'")  # 설명은 <description="..."> 따옴표 안에 들어간다

    model, tokenizer, snac = _models(ctx)
    device = ctx.device
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    chunks = split_text(text)
    ctx.log(f"{len(text)} chars in {len(chunks)} chunks")

    state = {"chunk": 0}

    class Cancel(StoppingCriteria):
        """새 토큰마다 취소를 보고 진행률을 알린다(조각 안에서는 최대 토큰 수 대비)."""

        def __call__(self, input_ids, scores, **kwargs):
            ctx.check_cancel()
            made = input_ids.shape[1] - state["prompt"]
            if made % 70 == 0:
                frac = (state["chunk"] + min(1.0, made / MAX_NEW_TOKENS)) / len(chunks)
                ctx.progress(0.1 + 0.85 * frac, "generate")
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    pieces = []
    start = time.monotonic()
    try:
        for i, chunk in enumerate(chunks):
            state["chunk"] = i
            prompt = build_prompt(tokenizer, description, chunk)
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            state["prompt"] = enc["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    min_new_tokens=MIN_NEW_TOKENS,
                    temperature=float(settings["temperature"]),
                    top_p=0.9,
                    repetition_penalty=1.1,
                    do_sample=True,
                    eos_token_id=CODE_END_TOKEN_ID,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([Cancel()]),
                )
            ids = out[0, state["prompt"]:].tolist()
            if CODE_END_TOKEN_ID in ids:
                ids = ids[:ids.index(CODE_END_TOKEN_ID)]
            codes = [t for t in ids if SNAC_MIN_ID <= t <= SNAC_MAX_ID]
            if len(codes) < SNAC_TOKENS_PER_FRAME:
                ctx.log(f"chunk {i}: no audio tokens", "warn")
                continue
            levels = unpack_snac_from_7(codes)
            with torch.inference_mode():
                z_q = snac.quantizer.from_codes([torch.tensor(l, dtype=torch.long, device=device).unsqueeze(0) for l in levels])
                audio = snac.decoder(z_q)[0, 0].float().cpu().numpy()
            if len(audio) > WARMUP_SAMPLES:
                audio = audio[WARMUP_SAMPLES:]
            pieces.append(audio.astype(np.float32))
            pieces.append(np.zeros(int(SAMPLE_RATE * 0.15), dtype=np.float32))  # 조각 사이 짧은 쉼
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise RuntimeError("그래픽카드 메모리가 모자랍니다. 다른 AI 작업 · 프로그램을 닫고 다시 하거나, 글을 줄여 주세요.") from None
    took = time.monotonic() - start
    if not pieces:
        raise RuntimeError("목소리가 만들어지지 않았습니다. 목소리 설명이나 글을 바꿔 다시 해 주세요.")
    audio = np.concatenate(pieces[:-1])
    out_path = ctx.out_path("maya1.wav")
    sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
    peak = torch.cuda.max_memory_reserved() / 2**20 if device == "cuda" else 0
    ctx.log(f"{len(chunks)} chunks → {audio.size / SAMPLE_RATE:.1f}s audio in {took:.1f}s, peak VRAM reserved {peak:.0f} MiB")
    ctx.progress(1.0, "done")
    return {"audio": out_path}


NODES = {"maya1": maya1}
