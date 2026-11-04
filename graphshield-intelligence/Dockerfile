FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY graphshield-data-engine /src/graphshield-data-engine
COPY graphshield-model-lab /src/graphshield-model-lab
COPY graphshield-intelligence /src/graphshield-intelligence
RUN pip install --no-cache-dir /src/graphshield-data-engine /src/graphshield-model-lab \
    /src/graphshield-intelligence
EXPOSE 8000
ENTRYPOINT ["graphshield-serve", "--host", "0.0.0.0", "--bundle", "/models/model.pt"]
