FROM aarri-agent-base:ubuntu24-pytest-tmux-yaml
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
WORKDIR /app
COPY paper_methods.md /app/paper_methods.md
COPY results_table.md /app/results_table.md
RUN mkdir -p /app/configs
COPY configs/das_bert_config.yaml /app/configs/das_bert_config.yaml
COPY configs/adam_config.yaml /app/configs/adam_config.yaml
COPY configs/sgd_config.yaml /app/configs/sgd_config.yaml
