"""detection_modes: each model's box dialect, the P(yes) gate, ownership and the verdict text.

  python3 -m unittest discover -s nvr/feed/tests
"""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import detection_modes as dm  # noqa: E402


class BoxDialects(unittest.TestCase):
    def test_qwen_bbox_2d_is_x_first(self):
        self.assertEqual(dm.parse_box('{"bbox_2d": [100, 200, 300, 400]}'), [0.1, 0.2, 0.3, 0.4])

    def test_gemma_box_2d_is_y_first(self):
        self.assertEqual(dm.parse_box('{"box_2d": [200, 100, 400, 300]}'), [0.1, 0.2, 0.3, 0.4])

    def test_internvl_bare_nested_list(self):
        self.assertEqual(dm.parse_box("the dog[[100, 200, 300, 400]]"), [0.1, 0.2, 0.3, 0.4])

    def test_degenerate_or_missing_box_is_none(self):
        self.assertIsNone(dm.parse_box('{"bbox_2d": [300, 200, 100, 400]}'))
        self.assertIsNone(dm.parse_box("no"))
        self.assertIsNone(dm.parse_box(""))

    def test_family_from_the_shims_model_id(self):
        self.assertEqual(dm.family("Qwen/Qwen3-VL-2B-Instruct (INT4)"), "qwen")
        self.assertEqual(dm.family("google/gemma-4-E2B-it (llama.cpp Q4_K_S)"), "gemma")
        self.assertEqual(dm.family("OpenGVLab/InternVL3_5-2B (INT4)"), "internvl")
        self.assertEqual(dm.family("nvidia/LocateAnything-3B (INT4, PyTorch)"), "locateanything")
        self.assertEqual(dm.family("nvidia/NVIDIA-Nemotron-3-Nano-4B (llama.cpp Q4_K_M, text only)"), "text")


class Gate(unittest.TestCase):
    def test_p_yes_sums_yes_spellings_in_top_logprobs(self):
        choice = {"logprobs": {"content": [{"token": "Yes", "logprob": math.log(0.6), "top_logprobs": [
            {"token": "Yes", "logprob": math.log(0.6)}, {"token": " yes", "logprob": math.log(0.1)},
            {"token": "No", "logprob": math.log(0.3)}]}]}}
        self.assertAlmostEqual(dm.p_yes(choice), 0.7, places=3)

    def test_no_logprobs_is_none(self):
        self.assertIsNone(dm.p_yes({"message": {"content": "yes"}}))

    def _check(self, answers, model="Qwen/Qwen3-VL-2B-Instruct (INT4)"):
        replies = iter(answers)
        def chat(url, image, text, max_tokens, logprobs, timeout):
            return next(replies)
        with mock.patch.object(dm, "model_id", return_value=model), mock.patch.object(dm, "_chat", chat):
            return dm.check("http://shim", b"jpeg", "bernese", min_confidence=0.5)

    def test_below_threshold_is_not_found_and_never_grounds(self):
        no = {"message": {"content": "no"}, "logprobs": {"content": [{"token": "no", "logprob": math.log(0.8),
              "top_logprobs": [{"token": "no", "logprob": math.log(0.8)}, {"token": "yes", "logprob": math.log(0.2)}]}]}}
        v = self._check([no])
        self.assertFalse(v["found"])
        self.assertAlmostEqual(v["confidence"], 0.2, places=3)
        self.assertEqual(len(v["answers"]), 1)

    def test_yes_then_box(self):
        yes = {"message": {"content": "yes"}, "logprobs": {"content": [{"token": "yes", "logprob": math.log(0.9),
               "top_logprobs": [{"token": "yes", "logprob": math.log(0.9)}]}]}}
        box = {"message": {"content": '{"bbox_2d": [100, 600, 300, 900]}'}}
        v = self._check([yes, box])
        self.assertTrue(v["found"])
        self.assertEqual(v["box"], [0.1, 0.6, 0.3, 0.9])
        self.assertEqual(dm.headline("bernese", v), "Bernese mountain dog spotted in the lower left of the frame (90% sure).")

    def test_locateanything_answers_in_one_step(self):
        ans = {"message": {"content": 'Found 1 Bernese mountain dog.\n[{"label": "Bernese mountain dog", "bbox_2d": [700, 100, 900, 300]}]'}}
        v = self._check([ans], model="nvidia/LocateAnything-3B (INT4, PyTorch)")
        self.assertTrue(v["found"])
        self.assertEqual(v["box"], [0.7, 0.1, 0.9, 0.3])
        self.assertIsNone(v["confidence"])

    def test_only_validated_engines_are_trusted(self):
        yes = {"message": {"content": "yes"}, "logprobs": {"content": [{"token": "yes", "logprob": 0.0}]}}
        box = {"message": {"content": '{"bbox_2d": [100, 600, 300, 900]}'}}
        self.assertTrue(self._check([yes, box])["engine_trusted"])
        self.assertFalse(self._check([yes, box], model="nvidia/Cosmos3-Edge (INT4 v3)")["engine_trusted"])

    def test_text_only_model_is_an_error_not_a_miss(self):
        v = self._check([], model="nvidia/NVIDIA-Nemotron-3-Nano-4B (llama.cpp Q4_K_M, text only)")
        self.assertFalse(v["found"])
        self.assertIn("text only", v["error"])


class Breed(unittest.TestCase):
    """The breed service's way: Frigate's boxes scored, the best one is the verdict."""

    def scored(self, scores, top=None):
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"scores": scores, "top": top or [["Siberian husky", 0.4]] * len(scores), "ms": 120}
        boxes = [[0.1, 0.1, 0.2, 0.2], [0.4, 0.3, 0.55, 0.47], [0.7, 0.6, 0.9, 0.9]][:len(scores)]
        with mock.patch.object(dm.requests, "post", return_value=resp) as post:
            v = dm.check_breed("http://breed", b"jpeg", boxes, "bernese")
        return v, post.call_args.kwargs["json"]

    def test_the_best_scoring_dog_is_found_and_boxed(self):
        v, sent = self.scored([0.02, 0.97, 0.05])
        self.assertTrue(v["found"])
        self.assertEqual(v["box"], [0.4, 0.3, 0.55, 0.47])
        self.assertEqual(v["confidence"], 0.97)
        self.assertTrue(v["engine_trusted"])
        self.assertIn("EntleBucher", sent["classes"])     # the look-alike family counts together
        self.assertEqual(len(sent["boxes"]), 3)

    def test_no_dog_over_the_gate_is_not_found(self):
        v, _ = self.scored([0.09, 0.04])
        self.assertFalse(v["found"])
        self.assertIsNone(v["box"])
        self.assertEqual(v["confidence"], 0.09)

    def test_service_down_is_an_error_not_a_miss(self):
        with mock.patch.object(dm.requests, "post", side_effect=dm.requests.ConnectionError("refused")):
            v = dm.check_breed("http://breed", b"jpeg", [[0.1, 0.1, 0.2, 0.2]], "bernese")
        self.assertFalse(v["found"])
        self.assertIn("ConnectionError", v["error"])


class Settings(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(dm, "STATE_PATH", Path(self.dir.name) / "modes.json")
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.dir.cleanup()

    def test_config_defaults_then_runtime_file(self):
        defaults = {"bernese": {"cameras": ["reachy_mini"], "speak": True}}
        self.assertEqual(dm.owner("reachy_mini", "dog", defaults), "bernese")
        self.assertIsNone(dm.owner("reachy_mini", "person", defaults))
        self.assertIsNone(dm.owner("front_driveway", "dog", defaults))
        state = dm.load_state(defaults)
        state["bernese"]["cameras"] = ["front_driveway"]
        dm.save_state(state)
        self.assertIsNone(dm.owner("reachy_mini", "dog", defaults))
        self.assertEqual(dm.owner("front_driveway", "dog", defaults), "bernese")
        self.assertTrue(json.loads(dm.STATE_PATH.read_text())["bernese"]["speak"])

    def test_engine_url_comes_from_config_only(self):
        state = dm.load_state({"bernese": {"url": "http://second-box:8105"}})
        self.assertEqual(state["bernese"]["url"], "http://second-box:8105")
        state["bernese"]["url"] = "http://somewhere-else"
        dm.save_state(state)
        self.assertEqual(dm.load_state({"bernese": {"url": "http://second-box:8105"}})["bernese"]["url"],
                         "http://second-box:8105")
        self.assertEqual(dm.load_state()["bernese"]["url"], "")

    def test_mode_default_gate_is_its_own(self):
        self.assertEqual(dm.load_state()["bernese"]["min_confidence"], 0.6)
        self.assertEqual(dm.load_state({"bernese": {"min_confidence": 0.7}})["bernese"]["min_confidence"], 0.7)

    def test_ntfy_push_attaches_the_boxed_frame(self):
        cfg = {"provider": "ntfy", "ntfy": {"server": "https://ntfy.example", "topic": "t", "token": ""}}
        with mock.patch.object(dm.requests, "put") as put:
            dm.push(cfg, "Reachy Mini · Bernese mountain dog", "spotted.", b"jpeg", "")
        url = put.call_args.args[0]
        kw = put.call_args.kwargs
        self.assertEqual(url, "https://ntfy.example/t")
        self.assertEqual(kw["data"], b"jpeg")
        self.assertEqual(kw["headers"]["Message"], "spotted.")
        self.assertNotIn("Authorization", kw["headers"])

    def test_where_words(self):
        self.assertEqual(dm.where([0.4, 0.4, 0.6, 0.6]), "in the middle of the frame")
        self.assertEqual(dm.where([0.7, 0.1, 0.9, 0.3]), "in the upper right of the frame")
        self.assertEqual(dm.where([0.0, 0.4, 0.2, 0.6]), "on the left of the frame")
        self.assertEqual(dm.where(None), "somewhere in the frame")


if __name__ == "__main__":
    unittest.main()
