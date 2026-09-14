#!/usr/bin/env python3
"""Alert policy for porch-dad: decide what is worth waking a phone for.

The VLM is asked ONLY to describe what it sees. Every alert category lives here, in code.

Why: this 4B model reports back whatever the prompt enumerates. Measured on real frames -
    prompt said "delivery and service workers"   -> "delivery worker" on 9 frames (a resident)
    prompt said "a gate standing open"           -> "gate standing open" (the gate was closed)
    prompt said "a person or animal and what it is doing" -> "a person is walking" (empty driveway)
Enumerating the alert criteria to the model manufactures those alerts. So the prompt enumerates
nothing, and this module classifies the resulting free-form description.

Alert scope, deliberately narrow:
    1. a living thing present/moving (people, animals) - not wind, foliage, shadow, rain
    2. a vehicle
    3. a gate that is open or ajar
    4. a package delivered or on the ground
    5. exterior lighting changing state
Anything else is NOT an alert.
"""
from __future__ import annotations

import re

DESCRIBE_PROMPT = "Describe only what is visible in this image, in one short sentence."

# --------------------------------------------------------------------------- category patterns
PERSON = re.compile(r"\b(person|people|man|men|woman|women|child|children|boy|girl|someone|"
                    r"figure|individual)\b", re.I)
ANIMAL = re.compile(r"\b(dog|cat|coyote|deer|raccoon|skunk|fox|possum|opossum|squirrel|bird|"
                    r"animal|rabbit|bear)\b", re.I)
VEHICLE = re.compile(r"\b(car|cars|suv|truck|van|vehicle|sedan|motorcycle|scooter|bicycle|bike|"
                     r"pickup)\b", re.I)
GATE_OPEN = re.compile(r"\bgate\b[^.]{0,40}\b(open|ajar|standing open|swung)\b"
                       r"|\b(open|ajar)\b[^.]{0,20}\bgate\b", re.I)
PACKAGE = re.compile(r"\b(package|parcel|box|boxes|delivery box|carton)\b", re.I)
LIGHT_ON = re.compile(r"\b(lamps?|lights?|lighting)\b[^.]{0,30}\b(lit|on|glowing|illuminated)\b"
                      r"|\b(lit|glowing|illuminated)\b[^.]{0,20}\b(lamps?|lights?)\b", re.I)

# A frame the model itself reports as degraded. Measured overnight on front_driveway, where Ring
# drops resolution in low light and the stream streaks on the change:
#     "A blurry view from a moving vehicle showing a street and buildings"      -> label person
#     "A distorted, pixelated image of a dimly lit room with a person"          -> label person
#     "A black SUV is parked under a concrete overpass at night"                -> label person
# All three scored 0.70-0.72, barely over threshold, and none contained a person. When the caption
# says the image is degraded, nothing else in that caption is trustworthy either - including any
# person the model hallucinated into the artefact, and including the detector label, which is what
# actually fired these. So a degraded frame suppresses the whole event rather than one category.
# The degradation must describe the IMAGE, not a subject in it. That distinction is load-bearing:
#     "A distorted, pixelated image of a dimly lit room"  -> the frame is junk, suppress
#     "A blurry view from a moving vehicle"               -> the frame is junk, suppress
#     "A blurry person in a white shirt"                  -> a REAL person, seen imperfectly. KEEP.
# A first attempt matched the bare adjective and suppressed that third caption - a genuine person
# on pinky at 20:33. Requiring an image noun nearby is what separates "the picture is broken" from
# "the subject is indistinct".
_DEGRADE_WORD = (r"blurry|blurred|distorted|pixelated|pixellated|garbled|glitch\w*|smear\w*|"
                 r"streak\w*|corrupted|grainy|unclear|out of focus|low[- ]quality")
_IMAGE_NOUN = r"image|images|view|views|footage|frame|frames|picture|pictures|video|feed|photo"
DEGRADED = re.compile(
    rf"\b(?:{_DEGRADE_WORD})\b[^.]{{0,30}}\b(?:{_IMAGE_NOUN})\b"
    rf"|\b(?:{_IMAGE_NOUN})\b[^.]{{0,30}}\b(?:{_DEGRADE_WORD})\b", re.I)

# Clause boundaries. The model writes both "No people are visible. A black cat is walking..." and
# the comma-joined "No people are visible, no animals are visible, no vehicles are visible", so
# splitting on sentences alone is not enough.
CLAUSE_SPLIT = re.compile(r"[.;!?]|,(?=\s*(?:no|not|there (?:are|is) no|nobody|none)\b)", re.I)

# A clause that denies something rather than reporting it.
NEGATION = re.compile(r"\b(no|not|never|nobody|no one|none|without|absent|unoccupied|empty)\b",
                      re.I)

# Noise the model narrates that is never an alert: the camera itself, lens artefacts, architecture,
# wall colours, weather and foliage.
NOISE = re.compile(r"\b(security camera|camera('s)? (lens|perspective)|fisheye|bounding box|"
                   r"ceiling|wall|walls|archway|canopy|overhang|awning|lens|distort\w*|"
                   r"wind|breeze|shadow|cloud|rain|foliage|tree|trees|bush|shrub)\b", re.I)

# Motion words raise a living-thing sighting to "movement", which is what the policy cares about.
# "a silver SUV is parked", "a car parked under the carport" - static background, not an anomaly.
PARKED_VEHICLE = re.compile(
    r"\b(car|cars|suv|truck|van|vehicle|sedan|pickup)\b[^.]{0,40}\bparked\b"
    r"|\bparked\b[^.]{0,40}\b(car|cars|suv|truck|van|vehicle|sedan|pickup)\b", re.I)

MOTION = re.compile(r"\b(walk\w*|run\w*|approach\w*|enter\w*|leav\w*|exit\w*|climb\w*|open\w*|"
                    r"mov\w*|stand\w*|crouch\w*|reach\w*|carry\w*|ride|riding|driving)\b", re.I)


# Frigate's detector is a trained object model and is more reliable than the VLM for mere
# PRESENCE. It labelled one event "dog" that the caption missed entirely. Fuse it in rather than
# trusting free-form prose alone.
FRIGATE_LABEL_MAP = {
    "person": "person",
    "dog": "animal", "cat": "animal", "bird": "animal", "bear": "animal",
    "horse": "animal", "cow": "animal", "sheep": "animal",
    "car": "vehicle", "truck": "vehicle", "motorcycle": "vehicle", "bus": "vehicle",
}


# A category that comes ONLY from the detector label, with nothing in the caption to corroborate
# it, needs the detector to be confident. Measured over one night on front_driveway:
#     0.70  "A distorted, pixelated image of a dimly lit room"      label person   (no person)
#     0.71  "A black SUV is parked under a concrete overpass"       label person   (no person)
#     0.72  "A blurry view from a moving vehicle"                   label person   (no person)
# every uncorroborated false positive sat at 0.70-0.72, while every real sighting that the caption
# also described scored 0.74-0.84. Caption-corroborated categories are NOT gated - a caption that
# names the thing is its own evidence, at any score.
FUSION_MIN_SCORE = 0.75


def classify(description: str, frigate_label: str | None = None,
             score: float | None = None) -> dict:
    """Return the alert categories a description supports, plus a cleaned alert line.

    Conservative by design: a category fires only on positive evidence. Absent evidence is not an
    alert, because a home security system that cries wolf gets muted and then protects nothing.
    """
    d = (description or "").strip()
    if not d:
        return {"alert": False, "categories": [], "headline": "", "raw": description}

    # A frame the model calls blurry/distorted/pixelated is evidence about the FRAME, not the scene.
    # Suppress before any category is considered, detector label included.
    if DEGRADED.search(d):
        return {"alert": False, "categories": [], "moving": False, "degraded": True,
                "headline": headline(d, []), "raw": d}

    # Match categories CLAUSE BY CLAUSE, skipping any clause that is a negation.
    #
    # Matching the whole caption at once reads a word inside a denial as a sighting. Every one of
    # these real captions produced a false category that way:
    #     "No exterior lamps are lit."          -> LIGHT_ON matched "lamps are lit"
    #     "There are no vehicles visible."       -> VEHICLE matched "vehicles"
    #     "No people are visible."               -> PERSON matched "people"
    #     "no animals are visible"               -> ANIMAL matched "animals"
    # Four of six `lights` matches one night were "No exterior lamps are lit" - the exact opposite
    # of the thing being alerted on. The model volunteers these denials readily, so this is not an
    # edge case; splitting on sentence and comma boundaries and dropping negated clauses is.
    positive = " ".join(c for c in CLAUSE_SPLIT.split(d) if c and not NEGATION.search(c))

    cats: list[str] = []
    person_negated = not PERSON.search(positive) and bool(PERSON.search(d))
    if PERSON.search(positive):
        cats.append("person")
    if ANIMAL.search(positive):
        cats.append("animal")
    # A vehicle only counts when it is DOING something. These driveways always contain a parked
    # car, so firing on every mention of it would alert on the background on every single event -
    # measured: 3 false alarms in 23 frames. Scope is "movement of ... vehicles", so a parked car
    # is background unless Frigate itself raised a vehicle event (fused below).
    vehicle_seen = bool(VEHICLE.search(positive))
    vehicle_parked = vehicle_seen and bool(PARKED_VEHICLE.search(positive))
    if vehicle_seen and not vehicle_parked:
        cats.append("vehicle")
    if GATE_OPEN.search(positive):
        cats.append("gate-open")
    if PACKAGE.search(positive):
        cats.append("package")
    if LIGHT_ON.search(positive):
        cats.append("lights")

    # Detector label wins on presence; the caption can only add to it. It ADDS, and only adds -
    # it must never resurrect a category the caption explicitly ruled out.
    #
    # This was measured, not theorised. Fusing the label unconditionally made the parked-vehicle
    # rule above dead code in production: Frigate labels every driveway event "car", so the label
    # re-added "vehicle" immediately after the caption suppressed it. Over 35 stored events the
    # result was a 100% alert rate - every single event fired, including six frames whose own
    # caption said the cars were parked, and one reading "An empty driveway with a closed gate."
    #
    # The fusion still earns its place for the other direction: on one frame Frigate labelled a dog
    # the caption missed entirely, and on another it caught a person at the edge of frame. Both
    # were confirmed by eye. So the label may add what the caption failed to see; it may not
    # overrule what the caption actually saw.
    mapped = FRIGATE_LABEL_MAP.get((frigate_label or "").lower())
    if (mapped and mapped not in cats
            and not (mapped == "vehicle" and vehicle_parked)
            and not (mapped == "person" and person_negated)
            # uncorroborated by the caption, so require detector confidence
            and (score is None or score >= FUSION_MIN_SCORE)):
        cats.append(mapped)

    moving = bool(MOTION.search(d)) and ({"person", "animal"} & set(cats))

    return {
        "alert": bool(cats),
        "categories": cats,
        "moving": bool(moving),
        "headline": headline(d, cats),
        "raw": d,
    }


def headline(description: str, cats: list[str]) -> str:
    """Strip the narration the model adds around the fact, so the phone shows the fact.

    "A security camera captures a man in a black shirt walking up a brick path in a backyard"
    becomes "A man in a black shirt walking up a brick path."
    """
    d = description
    d = re.sub(r"^\s*(a |the )?security camera (captures|shows|views)\s*", "", d, flags=re.I)
    d = re.sub(r",?\s*(viewed|seen|as seen|captured)\s+(from|through|by)[^.]*", "", d, flags=re.I)
    d = re.sub(r",?\s*through a (curved |fisheye )?(security )?camera( lens)?", "", d, flags=re.I)
    d = re.sub(r"\s{2,}", " ", d).strip()
    if d and d[0].islower():
        d = d[0].upper() + d[1:]
    return d


def summarize(cats: list[str]) -> str:
    order = ["person", "animal", "vehicle", "gate-open", "package", "lights"]
    label = {"person": "Person", "animal": "Animal", "vehicle": "Vehicle",
             "gate-open": "Gate open", "package": "Package", "lights": "Lights"}
    return " · ".join(label[c] for c in order if c in cats) or "Nothing notable"
