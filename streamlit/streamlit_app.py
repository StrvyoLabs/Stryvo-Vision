"""
Stryvo Vision — Streamlit demo.

Hosts the video-captioning demo on Streamlit Community Cloud (which can't run Docker).
It reuses the EXACT pipeline from the container's main.py / styles.py — same models,
same prompts, same reasoning_effort — so behavior matches the judged image. The Docker
image itself is never modified; this is a parallel demo path.

Because it runs the pipeline directly, it can show the real neutral scene description
(richer than the formal-caption workaround the Docker demo uses).

Key comes from st.secrets["FIREWORKS_API_KEY"] on Cloud, or the FIREWORKS_API_KEY env
var locally. ffmpeg is provided via the repo-root packages.txt on Cloud.
"""

import asyncio
import os
import shutil
import sys
import tempfile

import streamlit as st

# Make the repo-root modules (main.py, styles.py) importable from this subfolder.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import main as pipeline  # noqa: E402  (path set above)

STYLE_META = [
    ("formal", "Formal", "objective, factual", "#5b9dff"),
    ("sarcastic", "Sarcastic", "dry, ironic", "#ffb74d"),
    ("humorous_tech", "Humorous · Tech", "jokes with tech refs", "#43d6a0"),
    ("humorous_non_tech", "Humorous · Everyday", "no jargon", "#ff7bb0"),
]
STYLE_COLOR = {k: c for k, _, _, c in STYLE_META}
STYLE_NAME = {k: n for k, n, _, c in STYLE_META}

EXAMPLES = {
    "🍂 Autumn boulevard": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "🐱 Orange kitten": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "💻 Office worker": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
}


def get_api_key() -> str:
    try:
        k = st.secrets.get("FIREWORKS_API_KEY", "")
    except Exception:  # no secrets.toml present at all
        k = ""
    return (k or os.getenv("FIREWORKS_API_KEY", "")).strip()


async def run_pipeline(url: str, styles: list, client):
    """Reuse main.py's stages: download -> frames -> vision -> concurrent styling."""
    tmp = tempfile.mkdtemp(prefix="stviz_")
    try:
        vpath = os.path.join(tmp, "video.mp4")
        fdir = os.path.join(tmp, "frames")
        os.makedirs(fdir, exist_ok=True)

        if not await pipeline.download_video(url, vpath):
            raise RuntimeError("Couldn't download the video from that URL (check it's a direct .mp4 link).")
        frames = await asyncio.to_thread(pipeline.extract_frames, vpath, fdir)
        if not frames:
            raise RuntimeError("ffmpeg extracted no frames — is this a valid video?")
        b64 = await asyncio.to_thread(pipeline.frames_to_budgeted_b64, frames)
        description = await pipeline.get_scene_description(client, b64, "demo")
        caps = await asyncio.gather(
            *[pipeline.generate_style_caption(client, s, description, "demo") for s in styles]
        )
        return description, dict(zip(styles, caps)), len(frames)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- UI
st.set_page_config(page_title="Stryvo Vision", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
      .cap-card {background:#1a2338;border:1px solid #263149;border-radius:12px;padding:16px 18px;margin-bottom:14px;}
      .cap-hd {font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;}
      .cap-body {font-size:15px;line-height:1.5;color:#e8edf7;}
      .desc-card {border-left:3px solid #5b9dff;background:#131a2b;border-radius:8px;padding:14px 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stryvo Vision 🎬")
st.caption("Watches a video clip and writes captions in four styles — same pipeline as the Docker image.")

api_key = get_api_key()
if not api_key:
    st.error(
        "No Fireworks API key found. On Streamlit Cloud, add `FIREWORKS_API_KEY` under "
        "**App settings → Secrets**. Locally, set the `FIREWORKS_API_KEY` env var."
    )

if "url" not in st.session_state:
    st.session_state.url = ""

st.write("**Try an example:**")
ex_cols = st.columns(len(EXAMPLES))
for col, (label, ex_url) in zip(ex_cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state.url = ex_url

with st.form("gen"):
    st.text_input("Video URL (direct .mp4 link)", key="url", placeholder="https://.../clip.mp4")
    st.write("**Caption styles**")
    cols = st.columns(2)
    selected = []
    for i, (key, name, desc, color) in enumerate(STYLE_META):
        with cols[i % 2]:
            if st.checkbox(f"{name} — {desc}", value=True, key=f"cb_{key}"):
                selected.append(key)
    submitted = st.form_submit_button("Describe & Caption", type="primary", use_container_width=True)

if submitted:
    url = st.session_state.url.strip()
    if not url:
        st.warning("Enter a video URL (or click an example).")
    elif not selected:
        st.warning("Pick at least one caption style.")
    elif not api_key:
        st.warning("Set your Fireworks API key first (see above).")
    else:
        client = pipeline.AsyncOpenAI(
            base_url=pipeline.FIREWORKS_BASE_URL, api_key=api_key,
            timeout=pipeline.REQUEST_TIMEOUT, max_retries=0,
        )
        try:
            with st.spinner("Downloading, sampling frames, and captioning… (~20–40s)"):
                description, captions, n_frames = asyncio.run(run_pipeline(url, selected, client))
            st.session_state.result = {
                "url": url, "description": description, "captions": captions,
                "styles": selected, "n_frames": n_frames,
            }
        except Exception as e:  # noqa: BLE001
            st.session_state.result = None
            st.error(f"Failed: {e}")

res = st.session_state.get("result")
if res:
    st.divider()
    st.video(res["url"])
    st.caption(f"Analyzed {res['n_frames']} keyframes with {pipeline.VISION_MODEL.split('/')[-1]}.")

    st.subheader("📝 What's in the video")
    st.markdown(f'<div class="desc-card">{res["description"]}</div>', unsafe_allow_html=True)

    st.subheader("Captions")
    for key in res["styles"]:
        color = STYLE_COLOR.get(key, "#6d8bff")
        cap = res["captions"].get(key, "(no caption)")
        st.markdown(
            f'<div class="cap-card"><div class="cap-hd" style="color:{color}">'
            f'{STYLE_NAME.get(key, key)}</div><div class="cap-body">{cap}</div></div>',
            unsafe_allow_html=True,
        )

st.divider()
st.caption("Runs the same pipeline (main.py / styles.py) as saieesh09/stryvo-vision:latest. "
           "The Docker image is unchanged; this is a hosted demo path.")
