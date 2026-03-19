FROM mambaorg/micromamba:1.5.6-focal-cuda-12.1.1

COPY --chown=$MAMBA_USER:$MAMBA_USER ./environment.yaml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && \
    micromamba clean --all --yes

ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /app
COPY ./run_xihe_inference.py .
COPY ./get_inits_cmems.py .
COPY ./get_inits_wind.py .
COPY ./model.py .
COPY ./utilities.py .
COPY ./xihe_forecast.py .
COPY ./s3_upload.py .
COPY ./generate_thumbnails.py .

CMD ["python", "run_xihe_inference.py"]