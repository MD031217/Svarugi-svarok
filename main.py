"""Run training as a script, or serve the API with uvicorn main:app."""

from app.main import app

__all__ = ["app"]

if __name__ == "__main__":
    from scripts.train import main

    main()
