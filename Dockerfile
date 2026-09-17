FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install $(python -c "import tomllib; print(' '.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
COPY . .
RUN pip install --no-deps .
CMD ["uvicorn", "dsl.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
