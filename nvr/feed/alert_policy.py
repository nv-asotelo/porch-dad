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


def classify(description: str, frigate_label: str | None = None) -> dict:
    """Return the alert categories a description supports, plus a cleaned alert line.

    Conservative by design: a category fires only on positive evidence. Absent evidence is not an
    alert, because a home security system that cries wolf gets muted and then protects nothing.
    """
    d = (description or "").strip()
    if not d:
        return {"alert": False, "categories": [], "headline": "", "raw": description}

    cats: list[str] = []
    if PERSON.search(d):
        cats.append("person")
    if ANIMAL.search(d):
        cats.append("animal")
    # A vehicle only counts when it is DOING something. These driveways always contain a parked
    # car, so firing on every mention of it would alert on the background on every single event -
    # measured: 3 false alarms in 23 frames. Scope is "movement of ... vehicles", so a parked car
    # is background unless Frigate itself raised a vehicle event (fused below).
    vehicle_seen = bool(VEHICLE.search(d))
    vehicle_parked = vehicle_seen and bool(PARKED_VEHICLE.search(d))
    if vehicle_seen and not vehicle_parked:
        cats.append("vehicle")
    if GATE_OPEN.search(d):
        cats.append("gate-open")
    if PACKAGE.search(d):
        cats.append("package")
    if LIGHT_ON.search(d):
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
    if mapped and mapped not in cats and not (mapped == "vehicle" and vehicle_parked):
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
