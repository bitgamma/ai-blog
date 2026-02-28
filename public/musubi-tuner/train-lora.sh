#!/bin/bash

# Musubi-Tuner LoRA Training Script
# This script automates the LoRA training workflow for musubi-tuner

set -e  # Exit on error

# ==================== CONFIGURATION ====================
# Edit these values for your setup

# Model paths (provide the paths to your model files)
DIT_MODEL=""  # Path to diffusion model (e.g., flux-2-klein-base-4b.safetensors)
VAE_MODEL=""  # Path to VAE model (e.g., ae.safetensors)
TEXT_ENCODER=""  # Path to text encoder model (e.g., model-00001-of-00002.safetensors)

# Project configuration
PROJECT_NAME=""  # Name for your project (e.g., "my-style")
MODEL_VERSION="klein-base-4b"  # Model version: "klein-base-4b" or "klein-base-9b"

# Training parameters
NETWORK_DIM=16
NETWORK_ALPHA=16
LEARNING_RATE=1e-4
MAX_EPOCHS=30
SAVE_EVERY_N=2
BATCH_SIZE=4
RESOLUTION=1024

# ======================================================

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
    export MIOPEN_FIND_MODE=FAST
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export TORCH_BLAS_PREFER_HIPBLASLT=1

    PROJECT_DIR="${HOME}/${PROJECT_NAME}"
    OUTPUT_DIR="${PROJECT_DIR}/output"
    WORKSPACE_DIR="${HOME}/musubi-tuner-workspace"
    cd "$WORKSPACE_DIR/musubi-tuner"

    # Ensure virtual environment is active
    if [ -z "$VIRTUAL_ENV" ]; then
        source .venv/bin/activate
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

# Setup musubi-tuner environment
setup_musubi_tuner() {
    log_info "Setting up musubi-tuner environment..."
    
    # Create workspace directory if it doesn't exist
    WORKSPACE_DIR="${HOME}/musubi-tuner-workspace"
    mkdir -p "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
    
    # Clone repository if it doesn't exist
    if [ ! -d "musubi-tuner" ]; then
        log_info "Cloning musubi-tuner repository..."
        git clone https://github.com/kohya-ss/musubi-tuner/
    fi
    
    cd musubi-tuner
    
    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log_info "Creating Python virtual environment..."
        uv venv --python 3.12
    fi
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Install musubi-tuner with AMD GPU support
    log_info "Installing musubi-tuner..."
    uv pip install -e . --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151
    
    # Install torchvision with AMD GPU support
    log_info "Installing torchvision..."
    uv pip install torchvision --extra-index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151
    
    log_info "musubi-tuner environment setup complete."
}

# Validate user inputs
validate_inputs() {
    log_info "Validating inputs..."
    
    # Check if project name is set
    if [ -z "$PROJECT_NAME" ]; then
        log_error "PROJECT_NAME is not set. Please set it in the configuration section."
        exit 1
    fi
    
    # Check if model version is valid
    if [ "$MODEL_VERSION" != "klein-base-4b" ] && [ "$MODEL_VERSION" != "klein-base-9b" ]; then
        log_error "MODEL_VERSION must be 'klein-base-4b' or 'klein-base-9b'."
        exit 1
    fi
    
    # Check if model files are set
    if [ -z "$DIT_MODEL" ]; then
        log_error "DIT_MODEL is not set. Please set it in the configuration section."
        exit 1
    fi
    
    if [ -z "$VAE_MODEL" ]; then
        log_error "VAE_MODEL is not set. Please set it in the configuration section."
        exit 1
    fi
    
    if [ -z "$TEXT_ENCODER" ]; then
        log_error "TEXT_ENCODER is not set. Please set it in the configuration section."
        exit 1
    fi
    
    # Check if model files exist
    if [ ! -f "$DIT_MODEL" ]; then
        log_error "DIT_MODEL not found: $DIT_MODEL"
        exit 1
    fi
    
    if [ ! -f "$VAE_MODEL" ]; then
        log_error "VAE_MODEL not found: $VAE_MODEL"
        exit 1
    fi
    
    if [ ! -f "$TEXT_ENCODER" ]; then
        log_error "TEXT_ENCODER not found: $TEXT_ENCODER"
        exit 1
    fi
    
    log_info "All inputs validated successfully."
}

# Create project directories
create_project_dirs() {
    log_info "Creating project directories..."
    
    PROJECT_DIR="${HOME}/${PROJECT_NAME}"
    DATASET_DIR="${PROJECT_DIR}/dataset"
    CACHE_DIR="${PROJECT_DIR}/cache"
    OUTPUT_DIR="${PROJECT_DIR}/output"
    
    mkdir -p "$DATASET_DIR"
    mkdir -p "$CACHE_DIR"
    mkdir -p "$OUTPUT_DIR"
    
    log_info "Project directories ready:"
    log_info "  Project: ${PROJECT_DIR}"
    log_info "  Dataset: ${DATASET_DIR}"
    log_info "  Cache: ${CACHE_DIR}"
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
resolution = [${RESOLUTION}, ${RESOLUTION}]
caption_extension = ".txt"
batch_size = ${BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "${DATASET_DIR}"
cache_directory = "${CACHE_DIR}"
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
a portrait of a person with distinctive features, cinematic lighting
a landscape scene with natural beauty, vibrant colors
an object or scene of your choice, detailed rendering
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
    
    cat > "${PROJECT_DIR}/training.toml" << EOF
[general]
model_version = "${MODEL_VERSION}"
dit = "${DIT_MODEL}"
vae = "${VAE_MODEL}"
text_encoder = "${TEXT_ENCODER}"
dataset_config = "${PROJECT_DIR}/dataset.toml"
persistent_data_loader_workers = true
max_data_loader_n_workers = 2
compile = true
compile_mode = "default"

[network]
network_module = "networks.lora_flux_2"
network_dim = ${NETWORK_DIM}
network_alpha = ${NETWORK_ALPHA}

[optimizer]
optimizer_type = "AdamW"
learning_rate = ${LEARNING_RATE}

[training]
seed = 42
max_train_epochs = ${MAX_EPOCHS}
save_every_n_epochs = ${SAVE_EVERY_N}
mixed_precision = "bf16"
sdpa = true
timestep_sampling = "flux2_shift"
weighting_scheme = "none"

[output]
output_dir = "${OUTPUT_DIR}"
output_name = "${PROJECT_NAME}"
sample_prompts = "${PROJECT_DIR}/reference_prompts.txt"
sample_every_n_epochs = ${SAVE_EVERY_N}
sample_at_first = true
EOF

    log_info "Training config created at ${PROJECT_DIR}/training.toml"
}

# Cache latents
cache_latents() {
    log_info "Initializing environment..."
    init_env
    
    log_info "Caching latents..."
    
    python flux_2_cache_latents.py \
        --dataset_config "${PROJECT_DIR}/dataset.toml" \
        --vae "${VAE_MODEL}" \
        --model_version "${MODEL_VERSION}" \
        --disable_cudnn_backend
    
    log_info "Latents cached successfully."
}

# Cache text encoder outputs
cache_text_encoders() {
    log_info "Initializing environment..."
    init_env
    
    log_info "Caching text encoder outputs..."
    
    python flux_2_cache_text_encoder_outputs.py \
        --dataset_config "${PROJECT_DIR}/dataset.toml" \
        --text_encoder "${TEXT_ENCODER}" \
        --batch_size 16 \
        --model_version "${MODEL_VERSION}"
    
    log_info "Text encoder outputs cached successfully."
}

# Train the LoRA
train_lora() {
    log_info "Initializing environment..."
    init_env
    
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
    
    ACCELERATE_ARGS="--num_cpu_threads_per_process 1 --mixed_precision bf16 flux_2_train_network.py --config_file ${PROJECT_DIR}/training.toml --save_state"
    
    if [ -n "$RESUME_PATH" ]; then
        log_info "Resuming from checkpoint: ${RESUME_PATH}"
        ACCELERATE_ARGS="$ACCELERATE_ARGS --resume $RESUME_PATH"
    fi
    
    accelerate launch $ACCELERATE_ARGS
    
    log_info "Training completed!"
    log_info "Your LoRA checkpoints are in: ${OUTPUT_DIR}"
}

# Main execution
main() {
    echo "========================================"
    echo "  Musubi-Tuner LoRA Training Script"
    echo "========================================"
    echo ""
    
    check_dependencies
    setup_musubi_tuner
    validate_inputs
    create_project_dirs
    create_dataset_config
    create_reference_prompts
    create_training_config
    
    echo ""
    echo "========================================"
    echo "  Setup Complete!"
    echo "========================================"
    echo ""
    echo "Next steps:"
    echo "1. Add your training images to: ${DATASET_DIR}"
    echo "2. Edit captions for your images"
    echo "3. Edit reference prompts in: ${PROJECT_DIR}/reference_prompts.txt"
    echo "4. Run the caching and training steps:"
    echo ""
    echo "   bash ${PROJECT_DIR}/train-lora.sh cache"
    echo "   bash ${PROJECT_DIR}/train-lora.sh train"
    echo ""
}

# Run specific step
run_step() {
    case "$1" in
        cache)
            cache_latents
            cache_text_encoders
            ;;
        train)
            train_lora
            ;;
        *)
            log_error "Unknown step: $1"
            echo "Usage: $0 {cache|train}"
            exit 1
            ;;
    esac
}

# Entry point
if [ "$1" == "cache" ] || [ "$1" == "train" ]; then
    run_step "$1"
else
    main
fi