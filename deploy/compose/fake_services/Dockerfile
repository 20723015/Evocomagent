FROM python:3.12-slim

WORKDIR /srv
COPY fake_openai.py fake_commerce.py __init__.py ./
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"

EXPOSE 8000 8080
CMD ["python", "-m", "fake_openai"]