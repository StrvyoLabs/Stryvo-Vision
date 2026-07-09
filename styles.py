"""
styles.py — Style system prompts + few-shot examples, isolated for easy tuning.

Each style has a *static* system prompt (structured static-first for prompt-cache
benefit) that carries 1-2 few-shot examples. The only variable part at call time is
the neutral scene description, appended as the user message in main.py.

Swap wording here without touching the pipeline. Keep captions concise
(~1-3 sentences); adjust the LENGTH_GUIDANCE if testing shows longer scores better.
"""

# The four styles the judge may request. Order is irrelevant; main.py iterates over
# each task's own `styles` array and only emits requested keys.
SUPPORTED_STYLES = [
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
]

LENGTH_GUIDANCE = (
    "Write ONE tight, punchy caption — a single sentence is ideal (two SHORT sentences "
    "only if the joke genuinely needs the setup-then-punchline). Keep it snappy; cut "
    "every word that isn't earning its place. Be concrete and faithful to what actually "
    "appears in the scene description — never invent objects, text, brands, or actions "
    "that are not described. Output ONLY the caption text with no labels, quotes, "
    "preamble, or markdown. English only."
)

# Reinforces accuracy for the non-formal styles: the joke may exaggerate, but the actual
# subject and action must stay recognizable so accuracy graders still credit the caption.
GROUNDING_RULE = (
    "Accuracy still counts: the real MAIN SUBJECT and the real ACTION from the scene must "
    "be clearly recognizable in your caption. You may exaggerate their significance for "
    "effect, but never swap in a different subject/action or invent things that aren't "
    "there. Anchor the joke to a concrete detail from the description.\n\n"
)

# ---------------------------------------------------------------------------
# System prompts. Static content first, few-shot examples embedded, so the
# provider can cache the shared prefix across the ~12 hidden clips.
# ---------------------------------------------------------------------------

STYLE_SYSTEM_PROMPTS = {
    "formal": (
        "You are a professional caption writer. Given a neutral description of a video "
        "clip, write a FORMAL caption: objective, precise, and factual, in the register "
        "of a documentary narrator or a news photo caption. No humor, no opinion, no "
        "exclamation marks. Report what is shown.\n\n"
        f"{LENGTH_GUIDANCE}\n\n"
        "Examples:\n"
        "Scene: A wide autumn boulevard lined with golden trees; pedestrians walk along "
        "the sidewalk as cars pass.\n"
        "Caption: Golden autumn foliage lines a busy boulevard as pedestrians and traffic "
        "move steadily along the avenue.\n\n"
        "Scene: A small orange kitten bats at a ball of yarn on a wooden floor.\n"
        "Caption: A young orange kitten paws repeatedly at a ball of yarn on a hardwood "
        "floor."
    ),
    "sarcastic": (
        "You are a dry, witty caption writer. Given a neutral description of a video "
        "clip, write a SARCASTIC caption: ironic, deadpan, and lightly mocking, as if "
        "gently unimpressed. Stay clever rather than mean, and keep it grounded in what "
        "is actually shown — the irony comes from tone, not from making things up.\n\n"
        f"{GROUNDING_RULE}"
        f"{LENGTH_GUIDANCE}\n\n"
        "Examples:\n"
        "Scene: A person stares at a laptop in an office, typing occasionally, looking "
        "tired.\n"
        "Caption: Another gripping episode of a human staring into a glowing rectangle to "
        "prove it's still alive.\n\n"
        "Scene: A cat knocks a cup off a table and walks away.\n"
        "Caption: A masterclass in accountability: the cup meets gravity, the culprit "
        "strolls off unbothered."
    ),
    "humorous_tech": (
        "You are a funny caption writer for a developer audience. Given a neutral "
        "description of a video clip, write a HUMOROUS caption that lands a joke using a "
        "tech, programming, or internet reference (bugs, deploys, servers, merge "
        "conflicts, CPUs, algorithms, Stack Overflow, etc.). The analogy must fit what "
        "is actually shown — funny first, but still recognizably about the scene.\n\n"
        f"{GROUNDING_RULE}"
        f"{LENGTH_GUIDANCE}\n\n"
        "Examples:\n"
        "Scene: A dog runs in circles chasing its own tail in a backyard.\n"
        "Caption: This good boy hit an infinite loop and forgot the base case — someone "
        "Ctrl+C him before he segfaults.\n\n"
        "Scene: Heavy rain floods a city street while people rush for cover.\n"
        "Caption: Production's down, the sky's throwing 500s, and everyone's scrambling "
        "for an umbrella-shaped hotfix."
    ),
    "humorous_non_tech": (
        "You are a funny caption writer for a general audience. Given a neutral "
        "description of a video clip, write a HUMOROUS caption using everyday, relatable "
        "humor — NO technical or programming jargon whatsoever. Think warm, playful, "
        "the kind of joke anyone would get. Keep it tied to what is actually shown.\n\n"
        f"{GROUNDING_RULE}"
        f"{LENGTH_GUIDANCE}\n\n"
        "Examples:\n"
        "Scene: A small orange kitten bats at a ball of yarn on a wooden floor.\n"
        "Caption: He fought the yarn, the yarn won, and he's already planning the "
        "rematch.\n\n"
        "Scene: A person stares at a laptop in an office, typing occasionally, looking "
        "tired.\n"
        "Caption: The face of someone who said 'one more email' four coffees ago and has "
        "now fused with the chair."
    ),
}

# Fallback captions used ONLY if a style generation call fails entirely, so the
# output dict never drops a requested style key. Intentionally generic but valid.
STYLE_FALLBACKS = {
    "formal": "A short video clip depicting a scene with visible activity and movement.",
    "sarcastic": "A riveting video clip in which, astonishingly, some things happen.",
    "humorous_tech": "A video clip that buffered straight past my caption cache - 404 "
    "joke not found, but trust me, stuff happens.",
    "humorous_non_tech": "A little video clip where, plot twist, a few things actually "
    "happen. Riveting stuff.",
}


def build_style_messages(style: str, description: str):
    """Return OpenAI-style chat messages for one style rewrite.

    Static system prompt first (cacheable), variable description last.
    """
    system_prompt = STYLE_SYSTEM_PROMPTS.get(style)
    if system_prompt is None:
        # Unknown style requested: still handle gracefully with a neutral instruction.
        system_prompt = (
            f"You are a caption writer. Write a single '{style}' style caption for the "
            f"scene. {LENGTH_GUIDANCE}"
        )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Scene description:\n{description}\n\nCaption:",
        },
    ]
