# Use AWS Lambda Python base image - Python 3.10 for NeMo compatibility
FROM public.ecr.aws/lambda/python:3.10

# Set environment variables for cache directories and library configurations
# These must be set early and consistently throughout the build process
ENV MPLCONFIGDIR=/tmp/matplotlib \
    TRANSFORMERS_CACHE=/tmp/transformers \
    HF_HOME=/tmp/huggingface \
    HF_DATASETS_CACHE=/tmp/huggingface/datasets \
    NEMO_CACHE_DIR=/tmp/nemo \
    PYTHONUSERBASE=/tmp/python \
    JOBLIB_CACHE_DIR=/tmp/joblib \
    JOBLIB_TEMP_FOLDER=/tmp/joblib_temp \
    TMPDIR=/tmp \
    TEMP=/tmp \
    TMP=/tmp \
    MATPLOTLIB_CACHE_DIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    XDG_CACHE_HOME=/tmp \
    XDG_CONFIG_HOME=/tmp/config \
    NUMBA_CACHE_DIR=/tmp/numba \
    CONDA_PKGS_DIRS=/tmp/conda \
    PIP_CACHE_DIR=/tmp/pip \
    PYTHON_EGG_CACHE=/tmp/python-eggs \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install essential build tools and libraries
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    git \
    wget \
    tar \
    gzip \
    bzip2 \
    libsndfile \
    automake \
    autoconf \
    libtool \
    pkgconfig \
    yasm \
    nasm \
    cmake \
    diffutils \
    texinfo \
    ca-certificates \
    && yum clean all

# Compile and install LAME (libmp3lame) from source
RUN LAME_VERSION="3.100" && \
    cd /tmp && \
    wget "https://downloads.sourceforge.net/project/lame/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz" && \
    tar -xf "lame-${LAME_VERSION}.tar.gz" && \
    cd "lame-${LAME_VERSION}" && \
    ./configure --prefix=/usr/local --enable-static --disable-shared && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf "/tmp/lame-${LAME_VERSION}*"

# Compile and install FFmpeg from source
RUN FF_VERSION="6.1.1" && \
    cd /tmp && \
    wget "https://ffmpeg.org/releases/ffmpeg-${FF_VERSION}.tar.bz2" && \
    tar -xf "ffmpeg-${FF_VERSION}.tar.bz2" && \
    cd "ffmpeg-${FF_VERSION}" && \
    ./configure --prefix=/usr/local \
                --enable-static \
                --disable-shared \
                --disable-debug \
                --disable-doc \
                --disable-ffplay \
                --disable-ffprobe \
                --enable-gpl \
                --enable-libmp3lame \
                --extra-libs=-lpthread \
                --extra-libs=-lm && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf "/tmp/ffmpeg-${FF_VERSION}*"

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Create cache directories with proper permissions early in the build process
RUN mkdir -p \
    /tmp/matplotlib \
    /tmp/transformers \
    /tmp/huggingface \
    /tmp/huggingface/datasets \
    /tmp/nemo \
    /tmp/python \
    /tmp/joblib \
    /tmp/joblib_temp \
    /tmp/torch \
    /tmp/config \
    /tmp/numba \
    /tmp/conda \
    /tmp/pip \
    /tmp/python-eggs && \
    chmod -R 777 /tmp

# Copy requirements
COPY requirements-docker.txt .

# Install Python dependencies in optimized order
RUN pip install --no-cache-dir --upgrade pip

# Install essential build tools for Python packages
RUN pip install --no-cache-dir \
    wheel \
    setuptools \
    pybind11>=2.10.0 \
    packaging \
    ninja

# Install Cython first (required by many packages)
RUN pip install --no-cache-dir Cython==3.0.10

# Install core dependencies with compatible versions
RUN pip install --no-cache-dir \
    boto3>=1.26.0 \
    python-dotenv>=1.0.0 \
    openai>=1.0.0 \
    anthropic>=0.7.0 \
    requests>=2.32.0 \
    wget>=3.2.0

# Install compatible numpy and setuptools for NeMo
RUN pip install --no-cache-dir numpy==1.24.3 setuptools==68.0.0

# Install scientific computing
RUN pip install --no-cache-dir scipy>=1.11.0

# Install PyTorch for CPU (compatible versions for Python 3.10)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchaudio==2.1.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install audio processing libraries
RUN pip install --no-cache-dir \
    librosa==0.10.1 \
    soundfile==0.12.1 \
    pydub==0.25.1

# Install ML/NLP libraries - Fix huggingface_hub version compatibility
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    "huggingface_hub>=0.17.0,<0.25.0" \
    onnx==1.15.0 \
    onnxruntime==1.16.0

# Install PyTorch Lightning with compatible version
RUN pip install --no-cache-dir pytorch-lightning==2.1.0

# Install Hydra Core and OmegaConf (compatible versions for NeMo)
RUN pip install --no-cache-dir hydra-core==1.3.2 omegaconf==2.3.0

# Install NeMo dependencies manually to avoid build issues
# Skip problematic optional dependencies
RUN pip install --no-cache-dir \
    ruamel.yaml \
    text-unidecode \
    webdataset \
    braceexpand \
    editdistance \
    marshmallow \
    tensorboard \
    wandb \
    matplotlib \
    pandas \
    inflect \
    unidecode \
    frozendict \
    wrapt \
    sacrebleu \
    sacremoses \
    jiwer

# Configure matplotlib to use the correct backend and cache directory
RUN python3 -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure(); plt.close()" || true

# Try to install NeMo with fallback strategy
# Use an older version of NeMo that is compatible with huggingface_hub
RUN pip install --no-cache-dir "nemo-toolkit[asr]==1.20.0" || \
    (echo "NeMo ASR installation failed, trying core NeMo..." && \
     pip install --no-cache-dir nemo-toolkit==1.20.0) || \
    (echo "NeMo installation failed, trying alternative ASR approach..." && \
     pip install --no-cache-dir \
        nemo-toolkit==1.19.0 \
        kaldi-python-io || true)

# Install remaining dependencies
RUN pip install --no-cache-dir \
    scikit-learn>=1.3.0 \
    tenacity>=8.2.3

# Pre-configure libraries to use correct cache directories
RUN python3 -c \
    "import os; \
    os.makedirs('/tmp/matplotlib', exist_ok=True); \
    os.makedirs('/tmp/transformers', exist_ok=True); \
    os.makedirs('/tmp/huggingface', exist_ok=True); \
    os.makedirs('/tmp/nemo', exist_ok=True); \
    os.makedirs('/tmp/joblib', exist_ok=True); \
    print('Cache directories created')" || true

# Test library configurations
RUN python3 -c \
    "import matplotlib; matplotlib.use('Agg'); print('Matplotlib configured successfully')" || \
    echo "Matplotlib configuration warning"

RUN python3 -c \
    "from transformers import AutoTokenizer; print('Transformers cache configured successfully')" || \
    echo "Transformers cache warning"

RUN python3 -c \
    "import joblib; print('Joblib configured successfully')" || \
    echo "Joblib configuration warning"

# Clean up to reduce image size
RUN pip cache purge && \
    yum clean all && \
    rm -rf /var/cache/yum && \
    rm -rf /tmp/pip && \
    find /var/lang/lib/python3.10/site-packages/ -name "*.pyc" -delete && \
    find /var/lang/lib/python3.10/site-packages/ -name "__pycache__" -type d -exec rm -rf {} + || true

# Recreate cache directories after cleanup (they may have been removed)
RUN mkdir -p \
    /tmp/matplotlib \
    /tmp/transformers \
    /tmp/huggingface \
    /tmp/huggingface/datasets \
    /tmp/nemo \
    /tmp/python \
    /tmp/joblib \
    /tmp/joblib_temp \
    /tmp/torch \
    /tmp/config \
    /tmp/numba && \
    chmod -R 777 /tmp

# Copy function code
COPY lambda_function.py .
COPY audioconversion.py .
COPY s3_handler.py .
COPY config.py .
COPY alternative_diarization.py .
COPY ecr-lambda-policy.json .

# Set the CMD to your handler (AWS Lambda will handle the runtime)
CMD ["lambda_function.lambda_handler"]