#!/bin/bash

# SD-Scripts LoRA Training Script
# This script automates the LoRA training workflow for sd-scripts
# Supports SDXL and Anima models

set -e  # Exit on error

# ==================== CONFIGURATION ====================
# Edit these values for your setup

# Install parameters
GFX_NAME="${GFX_NAME:-gfx1151}"
SD_SCRIPTS_INSTALL_DIR="${SD_SCRIPTS_INSTALL_DIR:-$HOME}"

# Model paths (provide the paths to your model files)
DIT_MODEL="${DIT_MODEL:-}"  # Path to base model
VAE_MODEL="${VAE_MODEL:-}"                # Path to VAE model (for Anima)
T5XXL_TOKENIZER="${T5XXL_TOKENIZER:-}"    # Path to T5-XXL tokenizer (for Anima, optional)
QWEN3_MODEL="${QWEN3_MODEL:-}"            # Path to Qwen3 model (for Anima)
CLIP_L_MODEL="${CLIP_L_MODEL:-}"          # Path to CLIP-L model (for SDXL)
CLIP_G_MODEL="${CLIP_G_MODEL:-}"          # Path to CLIP-G model (for SDXL)

# Project configuration
PROJECT_NAME="${PROJECT_NAME:-}"          # Name for your project (e.g: my-lora)
MODEL_VERSION="${MODEL_VERSION:-sdxl}"          # Model type: "sdxl" or "anima"

# Training parameters
NETWORK_DIM="${NETWORK_DIM:-32}"
NETWORK_ALPHA="${NETWORK_ALPHA:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
SAVE_EVERY_N="${SAVE_EVERY_N:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RESOLUTION="${RESOLUTION:-1024}"

# ======================================================

# Runtime vars

PROJECT_DIR="${PWD}/${PROJECT_NAME}"
DATASET_DIR="${PROJECT_DIR}/dataset"
OUTPUT_DIR="${PROJECT_DIR}/output"

TRAINING_SCRIPT=""
NETWORK_MODULE=""
EXTRA_TRAINING_CONFIG=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Initialize environment (export AMD GPU settings + ensure venv is active)
init_env() {
    # Check if project name is set
    if [ -z "$PROJECT_NAME" ]; then
        log_error "PROJECT_NAME is not set. Please set it in the configuration section."
        exit 1
    fi

    case "$MODEL_VERSION" in
        sdxl) 
            TRAINING_SCRIPT="sdxl_train_network.py"
            NETWORK_MODULE="networks.lora"
            EXTRA_TRAINING_CONFIG="--cache_latents --cache_text_encoder_outputs"
            ;;
        anima)
            TRAINING_SCRIPT="anima_train_network.py"
            NETWORK_MODULE="networks.lora_anima"
            EXTRA_TRAINING_CONFIG="--cache_latents --cache_text_encoder_outputs"
            ;;
        *)
            log_error "MODEL_VERSION must be 'sdxl' or 'anima'."
            exit 1
            ;;
    esac

    export MIOPEN_FIND_MODE=FAST
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export TORCH_BLAS_PREFER_HIPBLASLT=1

    # Ensure virtual environment is active
    if [ -z "$VIRTUAL_ENV" ]; then
        source "${SD_SCRIPTS_INSTALL_DIR}/sd-scripts/.venv/bin/activate"
    fi
}

# Check if required commands exist
check_dependencies() {
    log_info "Checking dependencies..."
    
    for cmd in git uv; do
        if ! command -v $cmd &> /dev/null; then
            log_error "$cmd is not installed. Please install it first."
            exit 1
        fi
    done
    
    log_info "All dependencies found."
}

# Setup sd-scripts environment
setup_sd_scripts() {
    check_dependencies

    log_info "Setting up sd-scripts environment..."
    
    # Create workspace directory if it doesn't exist
    mkdir -p "$SD_SCRIPTS_INSTALL_DIR"
    cd "$SD_SCRIPTS_INSTALL_DIR"
    
    # Clone repository if it doesn't exist
    if [ ! -d "sd-scripts" ]; then
        log_info "Cloning sd-scripts repository..."
        git clone https://github.com/kohya-ss/sd-scripts.git
    fi
    
    cd sd-scripts
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log_info "Creating Python virtual environment..."
        uv venv --python 3.12
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install PyTorch with AMD GPU support
    log_info "Installing PyTorch with AMD GPU support..."
    uv pip install torch torchvision torchaudio triton --extra-index-url "https://rocm.nightlies.amd.com/v2-staging/$GFX_NAME"
    
    # Install sd-scripts requirements
    log_info "Installing sd-scripts requirements..."
    uv pip install --upgrade -r requirements.txt --extra-index-url "https://rocm.nightlies.amd.com/v2-staging/$GFX_NAME"
    
    log_info "sd-scripts environment setup complete."
}

# Validate user inputs
validate_inputs() {
    log_info "Validating inputs..."
    
    # Check if model files are set based on model type
    if [ -z "$DIT_MODEL" ]; then
        log_error "DIT_MODEL is not set. Please set it in the configuration section."
        exit 1
    fi
    
    if [ ! -f "$DIT_MODEL" ]; then
        log_error "DIT_MODEL not found: $DIT_MODEL"
        exit 1
    fi

    case "$MODEL_VERSION" in
        sdxl)
            if [ -z "$CLIP_L_MODEL" ]; then
                log_error "CLIP_L_MODEL is not set for SDXL. Please set it in the configuration section."
                exit 1
            fi
            if [ -z "$CLIP_G_MODEL" ]; then
                log_error "CLIP_G_MODEL is not set for SDXL. Please set it in the configuration section."
                exit 1
            fi
            if [ ! -f "$CLIP_L_MODEL" ]; then
                log_error "CLIP_L_MODEL not found: $CLIP_L_MODEL"
                exit 1
            fi
            if [ ! -f "$CLIP_G_MODEL" ]; then
                log_error "CLIP_G_MODEL not found: $CLIP_G_MODEL"
                exit 1
            fi
            ;;
        anima)
            if [ -z "$VAE_MODEL" ]; then
                log_error "VAE_MODEL is not set for Anima. Please set it in the configuration section."
                exit 1
            fi
            if [ -z "$T5XXL_TOKENIZER" ]; then
                log_info "T5XXL_TOKENIZER is not set for Anima. Using default configuration."
            fi
            if [ -z "$QWEN3_MODEL" ]; then
                log_error "QWEN3_MODEL is not set for Anima. Please set it in the configuration section."
                exit 1
            fi
            if [ ! -f "$VAE_MODEL" ]; then
                log_error "VAE_MODEL not found: $VAE_MODEL"
                exit 1
            fi
            if [ ! -f "$QWEN3_MODEL" ]; then
                log_error "QWEN3_MODEL not found: $QWEN3_MODEL"
                exit 1
            fi
            ;;
    esac
    
    log_info "All inputs validated successfully."
}

# Create project directories
create_project_dirs() {
    log_info "Creating project directories..."
    
    mkdir -p "$DATASET_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    log_info "Project directories ready:"
    log_info "  Project: ${PROJECT_DIR}"
    log_info "  Dataset: ${DATASET_DIR}"
    log_info "  Output: ${OUTPUT_DIR}"
}

# Create dataset config
create_dataset_config() {
    log_info "Creating dataset configuration..."
    
    if [ -f "${PROJECT_DIR}/dataset.toml" ]; then
        log_info "Dataset config already exists at ${PROJECT_DIR}/dataset.toml, skipping creation"
        return
    fi
    
    cat > "${PROJECT_DIR}/dataset.toml" << EOF
[general]
caption_extension = ".txt"

[[datasets]]
resolution = ${RESOLUTION}
batch_size = ${BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = true

  [[datasets.subsets]]
  image_dir = "${DATASET_DIR}"
  num_repeats = 1
EOF

    log_info "Dataset config created at ${PROJECT_DIR}/dataset.toml"
}

# Create reference prompts
create_reference_prompts() {
    log_info "Creating reference prompts..."
    
    if [ -f "${PROJECT_DIR}/reference_prompts.txt" ]; then
        log_info "Reference prompts already exist at ${PROJECT_DIR}/reference_prompts.txt, skipping creation"
        return
    fi
    
    cat > "${PROJECT_DIR}/reference_prompts.txt" << EOF
# Add prompts one per line to create sample images. Add as many as you need but remember that it takes time to generate them.
# You will also want to add a few parameters at the end of each prompt (on the same line). Most important ones are:
# --w: image width (eg: --w 1024)
# --h: image height (eg: --h 1024)
# --d: the seed. Setting a fixed seed is a good idea to make samples more comparable to each other (eg: --d 42)
# --s: the number of steps. A number between 30-50 will work fine for Anima (e.g: --s 30)
EOF

    log_info "Reference prompts created at ${PROJECT_DIR}/reference_prompts.txt"
    log_warn "Please edit the prompts to match your desired style!"
}

# Create training config
create_training_config() {
    log_info "Creating training configuration..."
    
    if [ -f "${PROJECT_DIR}/training.toml" ]; then
        log_info "Training config already exists at ${PROJECT_DIR}/training.toml, skipping creation"
        return
    fi
    
    # Build model arguments based on model type
    MODEL_ARGS=""
    case "$MODEL_VERSION" in
        sdxl)
            MODEL_ARGS="pretrained_model_name_or_path = \"${DIT_MODEL}\"
clip_l = \"${CLIP_L_MODEL}\"
clip_g = \"${CLIP_G_MODEL}\""
            ;;
        anima)
            MODEL_ARGS="pretrained_model_name_or_path = \"${DIT_MODEL}\"
vae = \"${VAE_MODEL}\"
t5xxl = \"${T5XXL_TOKENIZER}\"
qwen3 = \"${QWEN3_MODEL}\""
            ;;
    esac
    
    cat > "${PROJECT_DIR}/training.toml" << EOF
[general]
${MODEL_ARGS}
dataset_config = "${PROJECT_DIR}/dataset.toml"
persistent_data_loader_workers = true
max_data_loader_n_workers = 2
compile = true
compile_mode = "default"

[network]
network_module = "${NETWORK_MODULE}"
network_dim = ${NETWORK_DIM}
network_alpha = ${NETWORK_ALPHA}

[optimizer]
optimizer_type = "AdamW"
learning_rate = ${LEARNING_RATE}

[training]
seed = 42
max_train_epochs = ${MAX_EPOCHS}
mixed_precision = "bf16"
sdpa = true
${EXTRA_TRAINING_CONFIG}

[output]
output_dir = "${OUTPUT_DIR}"
output_name = "${PROJECT_NAME}"
save_every_n_epochs = ${SAVE_EVERY_N}
save_state = true
sample_prompts = "${PROJECT_DIR}/reference_prompts.txt"
sample_every_n_epochs = ${SAVE_EVERY_N}
sample_at_first = true
EOF

    log_info "Training config created at ${PROJECT_DIR}/training.toml"
}

# Create sd-scripts project (directories + configs)
create() {
    log_info "Creating project..."
    init_env

    validate_inputs
    create_project_dirs
    create_dataset_config
    create_reference_prompts
    create_training_config
    
    log_info "LoRA training project created successfully at ${PROJECT_DIR}"

    echo "Next steps:"
    echo "1. Add your training images to: ${DATASET_DIR}"
    echo "2. Add captions for your images (.txt files)"
    echo "3. Edit reference prompts in: ${PROJECT_DIR}/reference_prompts.txt"
    echo "4. Run the  training:"
    echo ""
    echo "   $0 train"
    echo ""    
}

# Train the LoRA
train_lora() {
    log_info "Initializing environment..."
    init_env
    
    cd ${SD_SCRIPTS_INSTALL_DIR}/sd-scripts

    # Check for existing checkpoints to resume from
    RESUME_PATH=""
    if [ -d "$OUTPUT_DIR" ]; then
        # Use find to list directories, sort by modification time (oldest first) and take the last one
        RESUME_PATH=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name "${PROJECT_NAME}"*state -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
        
        if [ -n "$RESUME_PATH" ]; then
            log_info "Found checkpoint with highest sequence: ${RESUME_PATH}"
        fi
    fi

    log_info "Starting training..."

    ACCELERATE_ARGS="--num_cpu_threads_per_process 1 --mixed_precision bf16  ${TRAINING_SCRIPT} --config_file ${PROJECT_DIR}/training"
    
    if [ -n "$RESUME_PATH" ]; then
        log_info "Resuming from checkpoint: ${RESUME_PATH}"
        ACCELERATE_ARGS="$ACCELERATE_ARGS --resume $RESUME_PATH"
    fi
    
    accelerate launch $ACCELERATE_ARGS
    
    log_info "Training completed!"
    log_info "Your LoRA checkpoints are in: ${OUTPUT_DIR}"
}

# Help function
help() {
    echo "Usage: $0 {setup|create|train}"
    echo ""
    echo "Actions:"
    echo "  setup    Install sd-scripts environment"
    echo "  create   Create a new LoRA training project"
    echo "  train    Train the LoRA"
}

case "$1" in
    setup)
        setup_sd_scripts
        ;;
    create)
        create
        ;;
    train)
        train_lora
        ;;
    *)
        help
        exit 1
        ;;
esac