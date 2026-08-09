from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_dockerfile_installs_deploy_extras_and_runs_asgi():
    dockerfile = read("Dockerfile")

    assert 'pip install --no-cache-dir -e ".[serve,vectorstores,docs]"' in dockerfile
    assert "python -m canon.product.asgi" in dockerfile
    assert "--host 0.0.0.0" in dockerfile
    assert "--port ${PORT:-8000}" in dockerfile
    assert "python -m canon.product.server" not in dockerfile


def test_compose_uses_openrouter_cohere_qdrant_without_openai_key():
    compose = read("docker-compose.yml")

    assert "OPENROUTER_API_KEY" in compose
    assert "COHERE_API_KEY" in compose
    assert "QDRANT_URL" in compose
    assert "QDRANT_API_KEY" in compose
    assert "CANON_VECTORSTORE: qdrant" in compose
    assert "CANON_ALLOWED_ORIGINS" in compose
    assert "CANON_REQUIRE_AUTH" in compose
    assert "CANON_API_KEY" in compose
    assert "OPENAI_API_KEY" not in compose


def test_env_files_are_kept_out_of_docker_build_context():
    dockerignore = read(".dockerignore")

    assert ".env" in dockerignore
    assert ".env.*" in dockerignore
    assert "!.env.example" in dockerignore
    assert "data/raw/*" in dockerignore
    assert "!data/raw/unstructured_ai_infra_geo_risk_demo.json" in dockerignore
    assert "data/processed/*" in dockerignore
    assert "!data/processed/chunks_ai_infra_geo_risk_demo.json" in dockerignore
    assert "!data/processed/works_ai_infra_geo_risk_demo.json" in dockerignore


def test_example_env_documents_deploy_knobs_without_openai_key():
    example = read(".env.example")

    assert "OPENROUTER_API_KEY=" in example
    assert "COHERE_API_KEY=" in example
    assert "QDRANT_URL=" in example
    assert "QDRANT_API_KEY=" in example
    assert "CANON_ALLOWED_ORIGINS=" in example
    assert "CANON_REQUIRE_AUTH=false" in example
    assert "CANON_BASIC_AUTH_USER=" in example
    assert "CANON_BASIC_AUTH_PASSWORD=" in example
    assert "CANON_API_KEY=" in example
    assert "CANON_DATA_DIR=" in example
    assert "CANON_REPORTS_DIR=" in example
    assert "OPENAI_API_KEY" not in example


def test_compose_has_render_like_smoke_service_without_host_corpus_mounts():
    compose = read("docker-compose.yml")

    assert "canon-render-smoke:" in compose
    assert 'profiles: ["render-smoke"]' in compose
    smoke_section = compose.split("canon-render-smoke:", 1)[1]
    assert "8001:8000" in smoke_section
    assert "volumes:" not in smoke_section


def test_render_blueprint_uses_free_docker_with_prompted_secrets():
    blueprint = read("render.yaml")

    assert "runtime: docker" in blueprint
    assert "plan: free" in blueprint
    assert "dockerfilePath: ./Dockerfile" in blueprint
    assert "dockerContext: ." in blueprint
    assert "healthCheckPath: /health" in blueprint
    assert "autoDeployTrigger: checksPass" in blueprint
    assert "key: CANON_REQUIRE_AUTH" in blueprint
    assert 'value: "true"' in blueprint
    for key in [
        "CANON_BASIC_AUTH_USER",
        "CANON_BASIC_AUTH_PASSWORD",
        "CANON_API_KEY",
        "OPENROUTER_API_KEY",
        "COHERE_API_KEY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ]:
        assert f"key: {key}" in blueprint
    assert blueprint.count("sync: false") >= 7
    assert "OPENAI_API_KEY" not in blueprint
