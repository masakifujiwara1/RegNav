import json

from regnav.training.tracking import TrainingTracker, build_dataset_summary


class FakeMlflow:
    def __init__(self):
        self.params = []
        self.tags = []
        self.artifacts = []

    def set_tracking_uri(self, uri):
        pass

    def set_experiment(self, experiment):
        pass

    def start_run(self, **kwargs):
        pass

    def log_params(self, params):
        self.params.append(params)

    def set_tags(self, tags):
        self.tags.append(tags)

    def log_metrics(self, metrics, step):
        pass

    def log_artifact(self, path):
        self.artifacts.append(path)

    def end_run(self, **kwargs):
        pass


def _tracker(tmp_path, summary):
    return TrainingTracker(
        enabled=True,
        output_dir=tmp_path / "run",
        model_config={},
        training_config={},
        model_config_path=tmp_path / "model.yaml",
        training_config_path=tmp_path / "training.yaml",
        split="train",
        manifest_hash="manifest-hash",
        device="cpu",
        compile_enabled=False,
        git_revision=None,
        tracking_uri="file:///tmp/mlruns",
        experiment="regnav",
        run_name="dataset-test",
        dataset_summary=summary,
    )


def test_build_dataset_summary_counts_trajectory_and_sample_subsets():
    assert build_dataset_summary(
        family="vint",
        split="train",
        manifest_hash="manifest-hash",
        trajectory_subsets=["recon", "beonav", "recon"],
        sample_subsets=["recon", "recon", "beonav", "recon"],
    ) == {
        "dataset_family": "vint",
        "dataset_subsets": ["beonav", "recon"],
        "split": "train",
        "manifest_hash": "manifest-hash",
        "trajectories": {"beonav": 1, "recon": 2},
        "samples": {"beonav": 1, "recon": 3},
    }


def test_tracker_logs_dataset_tags_params_and_summary_artifact(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setattr(
        "regnav.training.tracking.importlib.import_module", lambda name: fake
    )
    summary = build_dataset_summary(
        family="vint",
        split="train",
        manifest_hash="manifest-hash",
        trajectory_subsets=["recon", "recon"],
        sample_subsets=["recon"] * 10,
    )
    tracker = _tracker(tmp_path, summary)
    tracker.output_dir.mkdir()
    tracker.model_config_path.write_text("model: test\n")
    tracker.training_config_path.write_text("epochs: 1\n")
    (tracker.output_dir / "metrics.jsonl").write_text("{}\n")

    with tracker:
        tracker.log_artifacts(final_epoch=1, best_loss=0.25)

    assert fake.tags == [
        {
            "split": "train",
            "manifest_hash": "manifest-hash",
            "dataset_family": "vint",
            "dataset_subsets": "recon",
            "device": "cpu",
            "compile_enabled": "False",
            "git_revision": "unknown",
        }
    ]
    assert fake.params == [
        {
            "data.recon.trajectories": "2",
            "data.recon.samples": "10",
        }
    ]
    summary_path = tracker.output_dir / "dataset-summary.json"
    assert json.loads(summary_path.read_text()) == summary
    assert str(summary_path) in fake.artifacts
