"""Reproduce selected upstream CI failures and compare unchanged inputs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import _InvariantMetricMatrix
from geomstats.geometry.open_hemisphere import OpenHemispheresProduct
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.exponential_barycenter import ExponentialBarycenter
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.test.random import RandomDataGenerator


def payload_for(kind, seed):
    gs.random.seed(seed)
    if kind == "hemisphere":
        n = 3 + seed % 3
        generator = RandomDataGenerator(OpenHemispheresProduct(n), amplitude=5.0)
        base = generator.random_point(3)
        direction = generator.random_tangent_vec(base)
        tangent = generator.random_tangent_vec(base)
        return {"n": n, "base": gs.to_numpy(base).tolist(),
                "direction": gs.to_numpy(direction).tolist(),
                "tangent": gs.to_numpy(tangent).tolist()}
    space = SpecialOrthogonal(3)
    if kind == "barycenter":
        return {"points": gs.to_numpy(space.random_point(5)).tolist()}
    n_points = 3 if kind == "exp_after_log" else 4
    return {"base": gs.to_numpy(space.random_point(n_points)).tolist(),
            "point": gs.to_numpy(space.random_point(n_points)).tolist()}


def evaluate(kind, payload):
    try:
        if kind == "hemisphere":
            space = OpenHemispheresProduct(payload["n"])
            base, direction, tangent = [gs.array(payload[key])
                                        for key in ["base", "direction", "tangent"]]
            transported = space.metric.parallel_transport(tangent, base, direction=direction)
            end = space.metric.exp(direction, base)
            result = space.is_tangent(transported, end, atol=1e-6)
            return {"passed": bool(gs.all(result)), "values": gs.to_numpy(result).tolist()}
        if kind == "barycenter":
            space = SpecialOrthogonal(3)
            points = gs.array(payload["points"])
            first = ExponentialBarycenter(space).fit(points).estimate_
            second = FrechetMean(space, method="adaptive").fit(points).estimate_
            atol = 1e-3
        else:
            space = SpecialOrthogonal(payload.get("n", 3), equip=False)
            space.equip_with_metric(_InvariantMetricMatrix, left=False)
            base, point = gs.array(payload["base"]), gs.array(payload["point"])
            if kind == "exp_after_log":
                first = space.metric.exp(space.metric.log(point, base), base)
                second = point
                atol = 1e-3
            else:
                first = space.metric.dist(point, base)
                second = space.metric.dist(base, point)
                atol = 1e-2
        return {"passed": bool(gs.allclose(first, second, atol=atol)),
                "first": gs.to_numpy(first).tolist(), "second": gs.to_numpy(second).tolist(),
                "max_error": float(np.max(gs.to_numpy(gs.abs(first - second))))}
    except Exception as error:
        return {"passed": False, "error_type": type(error).__name__,
                "error_message": str(error).split(":")[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    parser.add_argument("output")
    parser.add_argument("--replay")
    parser.add_argument("--input")
    parser.add_argument("--max-seeds", type=int, default=200)
    parser.add_argument("--source", default="")
    args = parser.parse_args()
    touched = set()

    def profile(frame, event, arg):
        if event == "call" and args.source and frame.f_code.co_filename.endswith(args.source):
            touched.add(frame.f_code.co_name)

    if args.input:
        source_record = json.loads(Path(args.input).read_text())
        payload = source_record.get("payload", source_record)
        sys.setprofile(profile)
        result = evaluate(args.kind, payload)
        sys.setprofile(None)
        assert not result["passed"], result
        record = {"kind": args.kind, "payload": payload, "result": result,
                  "touched_functions": sorted(touched)}
    elif args.replay:
        baseline = json.loads(Path(args.replay).read_text())
        sys.setprofile(profile)
        result = evaluate(args.kind, baseline["payload"])
        sys.setprofile(None)
        record = {**baseline, "result": result, "touched_functions": sorted(touched)}
        previous = baseline["result"]
        assert result["passed"] == previous["passed"] == False
        assert result.get("error_type") == previous.get("error_type")
        for key in ["first", "second", "values"]:
            if key in previous:
                np.testing.assert_allclose(result[key], previous[key], rtol=1e-10, atol=1e-10)
        assert result.get("error_message") == previous.get("error_message")
    else:
        record = None
        for seed in range(args.max_seeds):
            payload = payload_for(args.kind, seed)
            result = evaluate(args.kind, payload)
            if result["passed"]:
                continue
            if args.kind == "barycenter":
                if "antipodal" not in result.get("error_message", ""):
                    continue
            elif "error_type" in result:
                continue
            record = {"kind": args.kind, "seed": seed, "payload": payload,
                      "result": result, "touched_functions": sorted(touched)}
            sys.setprofile(profile)
            record["result"] = evaluate(args.kind, payload)
            sys.setprofile(None)
            record["touched_functions"] = sorted(touched)
            break
        if record is None:
            raise RuntimeError(f"No matching failure in {args.max_seeds} seeds")
    record["backend"] = gs.__name__
    record["autograd_linalg_imported"] = "geomstats._backend.autograd.linalg" in sys.modules
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "payload"}))


if __name__ == "__main__":
    main()
