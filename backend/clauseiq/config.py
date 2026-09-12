from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./clauseiq.db"
    llm_provider: str = "mock"
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str = ""
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b"
    llm_temperature: float = 1.0
    llm_top_p: float = 0.95
    retrieval_top_k: int = 8
    rerank_top_k: int = 5
    max_upload_mb: int = 10
    max_review_steps: int = 8
    max_request_bytes: int = 12 * 1024 * 1024
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    allow_model_calls: bool = True
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
settings = Settings()
