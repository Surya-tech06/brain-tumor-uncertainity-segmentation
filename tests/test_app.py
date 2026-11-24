import io

import numpy as np
import pytest

import app as webapp
from btseg.classical import ClassicalEngine
from btseg.synthetic import make_dataset


@pytest.fixture(scope="module")
def client():
    xs, ys, bs = make_dataset(8, seed=1)
    webapp._engine = ClassicalEngine(n_trees=8).fit(xs, ys, bs, per_image=600)
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_sample_returns_maps_and_stats(client):
    r = client.get("/api/sample?seed=3&noise=0.05")
    d = r.get_json()
    assert r.status_code == 200
    assert {"input", "segmentation", "entropy", "epistemic", "aleatoric"} <= set(d["images"])
    assert d["images"]["entropy"].startswith("data:image/png;base64,")
    assert 0 <= d["stats"]["mean_entropy_in_brain_bits"] <= 1


def test_upload_npy(client):
    img = make_dataset(1, seed=9)[0][0, 0]
    buf = io.BytesIO(); np.save(buf, img); buf.seek(0)
    r = client.post("/api/segment", data={"file": (buf, "slice.npy")}, content_type="multipart/form-data")
    assert r.status_code == 200 and "tumour_pixels" in r.get_json()["stats"]


def test_upload_rejects_bad_type_and_missing_file(client):
    r = client.post("/api/segment", data={"file": (io.BytesIO(b"x"), "a.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400
    assert client.post("/api/segment").status_code == 400
