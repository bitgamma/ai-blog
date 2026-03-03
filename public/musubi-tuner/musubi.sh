#!/bin/bash

# Musubi-Tuner LoRA Training Script
# This script automates the LoRA training workflow for musubi-tuner

set -e  # Exit on error

# ==================== CONFIGURATION ====================
# Edit these values for your setup

# Install parameters
GFX_NAME="${GFX_NAME:-gfx1151}"
MUSUBI_TUNER_INSTALL_DIR="${MUSUBI_TUNER_INSTALL_DIR:-$HOME}"

# Model paths (provide the paths to your model files)
DIT_MODEL="${DIT_MODEL:-}"  # Path to diffusion model (e.g., flux-2-klein-base-4b.safetensors)
VAE_MODEL="${VAE_MODEL:-}"  # Path to VAE model (e.g., ae.safetensors)
TEXT_ENCODER="${TEXT_ENCODER:-}"  # Path to text encoder model (e.g., model-00001-of-00002.safetensors)

# Project configuration
PROJECT_NAME="${PROJECT_NAME:-}"  # Name for your project (e.g: my-lora)
MODEL_VERSION="${MODEL_VERSION:-}"  # Model version: "klein-base-4b", "klein-base-9b", "z-image"

# Training parameters
NETWORK_DIM="${NETWORK_DIM:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
SAVE_EVERY_N="${SAVE_EVERY_N:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
RESOLUTION="${RESOLUTION:-1024}"

# ======================================================

# Runtime vars

PROJECT_DIR="${PWD}/${PROJECT_NAME}"
DATASET_DIR="${PROJECT_DIR}/dataset"
CACHE_DIR="${PROJECT_DIR}/cache"
OUTPUT_DIR="${PROJECT_DIR}/output"

TRAINING_SCRIPT=""
CACHE_LATENT_SCRIPT=""
CACHE_TEXT_ENCODER_SCRIPT=""
NETWORK_NAME=""
TIMESTEP_SAMPLING=""
EXTRA_TRAINING_CONFIG=""
EXTRA_CACHING_ARGS=""

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
        klein-base-4b|klein-base-9b) 
            TRAINING_SCRIPT="flux_2_train_network.py"
            CACHE_LATENT_SCRIPT="flux_2_cache_latents.py"
            CACHE_TEXT_ENCODER_SCRIPT="flux_2_cache_text_encoder_outputs.py"
            NETWORK_NAME="networks.lora_flux_2"
            TIMESTEP_SAMPLING="flux2_shift"
            EXTRA_TRAINING_CONFIG="model_version = \"${MODEL_VERSION}\""
            EXTRA_CACHING_ARGS="--model_version \"${MODEL_VERSION}\""
            ;;
        z-image)
            TRAINING_SCRIPT="zimage_train_network.py"
            CACHE_LATENT_SCRIPT="zimage_cache_latents.py"
            CACHE_TEXT_ENCODER_SCRIPT="zimage_cache_text_encoder_outputs.py"
            NETWORK_NAME="networks.lora_zimage"
            TIMESTEP_SAMPLING="shift"
            EXTRA_TRAINING_CONFIG="discrete_flow_shift = 2.0"
            ;;
        *)
            log_error "MODEL_VERSION must be 'klein-base-4b', 'klein-base-9b' or 'z-image'."
            exit 1
            ;;
    esac    

    export MIOPEN_FIND_MODE=FAST
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
    export TORCH_BLAS_PREFER_HIPBLASLT=1

    cd "$MUSUBI_TUNER_INSTALL_DIR/musubi-tuner"

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
    check_dependencies

    log_info "Setting up musubi-tuner environment..."
    
    # Create workspace directory if it doesn't exist
    mkdir -p "$MUSUBI_TUNER_INSTALL_DIR"
    cd "$MUSUBI_TUNER_INSTALL_DIR"
    
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
    uv pip install -e . --extra-index-url "https://rocm.nightlies.amd.com/v2-staging/$GFX_NAME"
    
    # Install torchvision with AMD GPU support
    log_info "Installing torchvision..."
    uv pip install torchvision --extra-index-url "https://rocm.nightlies.amd.com/v2-staging/$GFX_NAME"
    
    log_info "musubi-tuner environment setup complete."
}

# Validate user inputs
validate_inputs() {
    log_info "Validating inputs..."
    
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
# add prompts one per line to create sample images. Add as many as you need but remember that it takes time to generate them
# you will also want to add a few parameters at the end of each prompt (on the same line). Most important ones are:
# --w: image width (eg: --w 1024)
# --h: image height (eg: --h 1024)
# --d: the seed. Setting a fixed seed is a good idea to make samples more comparable to each other (eg: --d 42)
# --s: the number of steps. A number between and 30-50 will work fine for Klein (e.g: --s 30)
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
dit = "${DIT_MODEL}"
vae = "${VAE_MODEL}"
text_encoder = "${TEXT_ENCODER}"
dataset_config = "${PROJECT_DIR}/dataset.toml"
persistent_data_loader_workers = true
max_data_loader_n_workers = 2
compile = true
compile_mode = "default"

[network]
network_module = "${NETWORK_NAME}"
network_dim = ${NETWORK_DIM}

[optimizer]
optimizer_type = "AdamW"
learning_rate = ${LEARNING_RATE}

[training]
seed = 42
max_train_epochs = ${MAX_EPOCHS}
mixed_precision = "bf16"
sdpa = true
timestep_sampling = "${TIMESTEP_SAMPLING}"
weighting_scheme = "none"
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

# Create Musubi-Tuner project (directories + configs)
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
    echo "2. Edit captions for your images"
    echo "3. Edit reference prompts in: ${PROJECT_DIR}/reference_prompts.txt"
    echo "4. Run the caching and training steps:"
    echo ""
    echo "   $0 cache"
    echo "   $0 train"
    echo ""    
}

# Cache latents
cache_latents() {
    log_info "Initializing environment..."
    init_env
    
    log_info "Caching latents..."
    
    python "${CACHE_LATENT_SCRIPT}" \
        --dataset_config "${PROJECT_DIR}/dataset.toml" \
        --vae "${VAE_MODEL}" \
        --disable_cudnn_backend \
        ${EXTRA_CACHING_ARGS}
    
    log_info "Latents cached successfully."
}

# Cache text encoder outputs
cache_text_encoders() {
    log_info "Initializing environment..."
    init_env
    
    log_info "Caching text encoder outputs..."
    
    python "${CACHE_TEXT_ENCODER_SCRIPT}" \
        --dataset_config "${PROJECT_DIR}/dataset.toml" \
        --text_encoder "${TEXT_ENCODER}" \
        --batch_size 16 \
        ${EXTRA_CACHING_ARGS}
    
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

    ACCELERATE_ARGS="--num_cpu_threads_per_process 1 --mixed_precision bf16 ${TRAINING_SCRIPT} --config_file ${PROJECT_DIR}/training.toml"
    
    if [ -n "$RESUME_PATH" ]; then
        log_info "Resuming from checkpoint: ${RESUME_PATH}"
        ACCELERATE_ARGS="$ACCELERATE_ARGS --resume $RESUME_PATH"
    fi
    
    accelerate launch $ACCELERATE_ARGS
    
    log_info "Training completed!"
    log_info "Your LoRA checkpoints are in: ${OUTPUT_DIR}"
}

# Convert z-image LoRA for ComfyUI compatibility
convert_lora() {
    local INPUT_PATH="$1"
    local OUTPUT_PATH="$2"
    
    # Auto-detect latest checkpoint if not provided
    if [ -z "$INPUT_PATH" ]; then
        INPUT_PATH=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name "${PROJECT_NAME}*.safetensors" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
        
        if [ -z "$INPUT_PATH" ]; then
            log_error "No LoRA checkpoint found in ${OUTPUT_DIR}"
            log_info "Usage: $0 convert [checkpoint_path] [--output output_path]"
            return 1
        fi
        
        log_info "No checkpoint specified. Using latest: ${INPUT_PATH}"
    fi
    
    # Validate input exists
    if [ ! -f "$INPUT_PATH" ]; then
        log_error "Checkpoint not found: $INPUT_PATH"
        return 1
    fi
    
    # Auto-generate output path if not provided
    if [ -z "$OUTPUT_PATH" ]; then
        local BASENAME=$(basename "$INPUT_PATH" .safetensors)
        OUTPUT_PATH="${OUTPUT_DIR}/${BASENAME}_comfyui.safetensors"
    fi
    
    log_info "Converting z-image LoRA for ComfyUI compatibility..."
    log_info "Input:  ${INPUT_PATH}"
    log_info "Output: ${OUTPUT_PATH}"
    
    python "convert_lora.py" \
        --input "$INPUT_PATH" \
        --output "$OUTPUT_PATH" \
        --target "other"
    
    log_info "Conversion complete!"
}

# Help function
help() {
    echo "Usage: $0 {setup|create|cache|train|convert}"
    echo ""
    echo "Actions:"
    echo "  setup     Install musubi-tuner environment"
    echo "  create    Create a new LoRA training project"
    echo "  cache     Cache latents and text encoder outputs"
    echo "  train     Train the LoRA"
    echo "  convert   Convert z-image LoRA for ComfyUI"
    echo "            Usage: $0 convert [checkpoint] [--output path]"
}

case "$1" in
    setup)
        setup_musubi_tuner
        ;;
    create)
        create
        ;;
    cache)
        cache_latents
        cache_text_encoders
        ;;
    train)
        train_lora
        # Automatically convert z-image LoRA for ComfyUI
        if [ "$MODEL_VERSION" == "z-image" ] && [ -n "$CONVERSION_SCRIPT" ]; then
            convert_lora
        fi
        ;;
    convert)
        convert_lora "$2" "$3"
        ;;
    *)
        help
        exit 1
        ;;
esac