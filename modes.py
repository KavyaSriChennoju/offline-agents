"""
Shared bits for the modes. The modes themselves live in 4_agent.ipynb,
where you can edit them and re-run the cell.

---------------------------------------------------------------------------
Fields (all optional except name + system):

  name         shown on screen
  system       the system prompt. {now} is replaced with the current time.
  look         when to send a camera frame with your words
                 "always"        every turn
                 "when_asked"    only if you say see/look/this/read/holding...
                 "never"         blind
                 "before_after"  snapshot when the mode starts, then send BEFORE + NOW
                 "burst"         grab `burst` frames, `burst_gap` seconds apart (a tiny video)
  memory       how older turns are kept in the prompt
                 "latest_image"  only the current frame is an image; old turns are text only
                 "all_images"    keep every frame (remembers the most, fills the context fastest)
                 "captions"      after each turn, write a one-line caption of the frame
                                 and keep that instead of the image (cheap visual memory)
  max_turns    how many past exchanges to keep (older ones fall off)
  kickoff      a hidden instruction sent when the mode starts, so the bot speaks first
  hide         regex; matching lines stay in memory but are never shown or spoken.
               Handy for secrets the bot needs to remember (see i_spy).
  temperature, max_tokens   the usual
  burst, burst_gap          for look="burst"
---------------------------------------------------------------------------
"""

VOICE_RULES = (
    " Your reply is spoken aloud: one to three short sentences, no lists, no markdown, no emoji."
)

HONESTY = (
    " If something isn't visible in the image, say you can't see it. Never guess text you can't read."
)

# Words that count as "asking to look" for look="when_asked"
LOOK_WORDS = (
    "see", "look", "this", "these", "that", "holding", "hand", "show", "read", "wearing",
    "colour", "color", "behind", "screen", "camera", "here", "what's in", "how many", "count",
)
