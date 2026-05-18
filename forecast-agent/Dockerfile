FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY main.py openRouter.py serperSearch.py calibrate.py kalshi_prices.py market_prior.py \
    event_context.py research.py ./
COPY v3 ./v3

ENV PATH="/app/.venv/bin:$PATH"
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
