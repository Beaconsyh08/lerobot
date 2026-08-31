from pathlib import Path

from lerobot.data_platform import viewer


def _page_map(groups: list[dict]) -> dict[str, dict]:
    return {
        page["key"]: page
        for group in groups
        for page in group["pages"]
    }


def test_full_console_navigation_uses_workspace_page_tab_hierarchy():
    groups = viewer._console_groups_for_tabs(
        viewer._CONSOLE_MODE_ALLOWED_TABS[viewer.CONSOLE_MODE_FULL],
        allowed_open_links=viewer._CONSOLE_MODE_ALLOWED_OPEN_LINKS[viewer.CONSOLE_MODE_FULL],
        legacy_mutations_enabled=False,
    )

    assert [group["key"] for group in groups] == ["data_platform", "data_curation"]
    pages = _page_map(groups)
    assert list(pages) == [
        "datasets",
        "preprocessing",
        "versions",
        "runs",
        "explore",
        "quality",
        "annotation",
        "dataset_build",
    ]
    assert [tab["key"] for tab in pages["preprocessing"]["tabs"]] == [
        "cache",
        "standardize",
        "transform",
        "split_merge",
    ]
    assert [tab["key"] for tab in pages["explore"]["tabs"]] == [
        "explore_overview",
        "embedding",
        "compare",
    ]
    assert pages["explore"]["open_links"] == ["viewer", "analysis"]


def test_visualize_console_keeps_dataset_runs_explore_and_allowed_legacy_page():
    groups = viewer._console_groups_for_tabs(
        viewer._CONSOLE_MODE_ALLOWED_TABS[viewer.CONSOLE_MODE_VISUALIZE],
        allowed_open_links=viewer._CONSOLE_MODE_ALLOWED_OPEN_LINKS[viewer.CONSOLE_MODE_VISUALIZE],
        legacy_mutations_enabled=True,
    )

    pages = _page_map(groups)
    assert set(pages) == {
        "datasets",
        "preprocessing",
        "runs",
        "explore",
        "quality",
        "admin_operations",
    }
    assert [tab["key"] for tab in pages["preprocessing"]["tabs"]] == ["cache"]
    assert [tab["key"] for tab in pages["explore"]["tabs"]] == ["explore_overview"]
    assert [tab["key"] for tab in pages["admin_operations"]["tabs"]] == ["dataset_ops"]
    assert pages["explore"]["open_links"] == ["viewer", "analysis"]
    assert next(group for group in groups if group["key"] == "legacy_admin")["label"] == "Admin Mode"


def test_homepage_uses_single_canvas_dataset_page_and_on_demand_job_drawer():
    template = (
        Path(__file__).parents[2]
        / "lerobot"
        / "data_platform"
        / "templates"
        / "visualize_dataset_homepage.html"
    ).read_text()

    assert "dp-console-grid" not in template
    assert "Job & Artifacts" not in template
    assert "dp-main-canvas" in template
    assert 'x-show="activePage === \'datasets\'"' in template
    assert 'x-show="activePage === \'runs\'"' in template
    assert 'x-show="jobDrawerOpen"' in template
    assert "Manage datasets" in template
    assert "View all runs" in template
    assert "params.set('page', this.activePage)" in template
    assert "initialPage: {{ initial_page|tojson }}" in template
    assert "Load selected" not in template
    assert "Load registered" not in template
    assert "Register / Load" not in template
    assert "datasetSelection" not in template
    assert 'x-show="dataset.cache_only"' not in template
    assert 'x-show="candidate.cache_only"' not in template
    assert "Mark root as source" in template
    assert "source protected" in template
    assert "Source Delivery" in template
    assert "Dataset Requirement" in template
    assert "Deterministic Data Recipe" in template
    assert "Training & Collection Feedback" not in template
    assert "Enter Admin" in template
    assert "Admin active" in template
    assert "Admin Mode enabled" in template
    assert "dp-header-utilities" in template
    assert "adminMenuOpen" in template
    assert "Set administrator password" in template
    assert "/api/admin/setup" in template
    assert "/api/admin/login" in template
    assert "/api/admin/logout" in template
    assert "dataPlatform.adminModeUntil" not in template
    assert "legacyMutationsEnabled && adminModeEnabled" in template
    assert "Lifecycle stage" in template
    assert "registered or original input" in template
    assert "preprocessing output" in template
    assert "reviewed construction or Manifest output" in template
    assert "datasetStageClass(item)" in template
    assert "datasetStageHint(item)" in template
    assert 'x-model="preprocess.delete_reason"' in template
    assert 'x-model="preprocess.flag_delete_reason"' in template


def test_homepage_uses_consistent_visual_hierarchy_and_context_states():
    template = (
        Path(__file__).parents[2]
        / "lerobot"
        / "data_platform"
        / "templates"
        / "visualize_dataset_homepage.html"
    ).read_text()

    assert "dp-app-header" in template
    assert "dp-page-surface" in template
    assert "dp-page-kicker" in template
    assert "dp-context-card" in template
    assert "dp-empty-state" in template
    assert "Working dataset" in template
    assert "activeWorkspaceLabel()" in template
    assert "font-mono text-sm" not in template


def test_explore_overview_surfaces_visualizations_and_preparation_paths():
    template = (
        Path(__file__).parents[2]
        / "lerobot"
        / "data_platform"
        / "templates"
        / "visualize_dataset_homepage.html"
    ).read_text()

    assert "Visualization center" in template
    assert "visualizationItems()" in template
    assert "openVisualizationSetup(item.key)" in template
    for label in (
        "Episode Viewer",
        "Dataset Analysis",
        "Label Review",
        "Tag Review",
        "Construction Review",
        "Embedding Map",
        "Dataset Compare",
        "Smoothing Report",
    ):
        assert label in template
    assert ".filter(item => this.openLinkEnabled(item.key))" in template


def test_robot_profile_and_viewer_signal_layout_are_presented_separately():
    templates_dir = (
        Path(__file__).parents[2] / "lerobot" / "data_platform" / "templates"
    )
    homepage = (templates_dir / "visualize_dataset_homepage.html").read_text()
    viewer_template = (templates_dir / "visualize_dataset_template.html").read_text()

    assert "Robot / Stage profile" in homepage
    assert "Output signal schema is Standard 16D" in homepage
    assert "hasBodyJoints" in viewer_template
    assert 'data_version == "DVT2"' not in viewer_template
