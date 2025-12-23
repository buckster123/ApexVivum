#!/bin/bash

# ==============================================================================
# Apex Aurum - Raspberry Pi 5 Installation Script
# Python 3.13.5, ARM64, Virtual Environment with System Site Packages
# ==============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.13.5"
VENV_NAME="apex-venv"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${PROJECT_DIR}/${VENV_NAME}"
SANDBOX_DIR="${PROJECT_DIR}/sandbox"

# Check if running on Raspberry Pi
check_pi() {
    if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
        echo -e "${YELLOW}Warning: This script is optimized for Raspberry Pi. Continuing anyway...${NC}"
    fi
}

# Print colored status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python_version() {
    if ! command_exists python3; then
        print_error "Python3 is not installed!"
        exit 1
    fi
    
    local version
    version=$(python3 --version 2>&1 | awk '{print $2}')
    print_status "Found Python version: $version"
    
    if [[ "$version" != "$PYTHON_VERSION" ]]; then
        print_warning "Expected Python $PYTHON_VERSION, but found $version"
        print_status "Attempting to install Python $PYTHON_VERSION..."
        return 1
    fi
    return 0
}

# Install Python 3.13.5 from source if needed
install_python_from_source() {
    print_status "Installing Python $PYTHON_VERSION from source (this may take 20-30 minutes)..."
    
    sudo apt update
    sudo apt install -y build-essential zlib1g-dev libncurses5-dev \
        libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev \
        libsqlite3-dev wget libbz2-dev liblzma-dev
    
    cd /tmp
    wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz"
    tar -xf "Python-${PYTHON_VERSION}.tar.xz"
    cd "Python-${PYTHON_VERSION}"
    
    ./configure --enable-optimizations --enable-shared \
        --with-system-ffi --with-system-expat \
        LDFLAGS="-Wl,--strip-all"
    
    make -j$(nproc)
    sudo make altinstall
    
    cd /tmp
    rm -rf "Python-${PYTHON_VERSION}" "Python-${PYTHON_VERSION}.tar.xz"
    
    print_success "Python $PYTHON_VERSION installed successfully"
}

# Update system packages
update_system() {
    print_status "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    print_success "System packages updated"
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Core build tools and Python development
    sudo apt install -y \
        build-essential \
        python3-dev \
        python3-pip \
        python3-venv \
        pkg-config \
        cmake \
        git \
        curl \
        wget
    
    # Numerical and scientific libraries
    # FIXED: Replaced deprecated libatlas-base-dev with libopenblas-dev
    sudo apt install -y \
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        gfortran \
        libhdf5-dev
    
    # Database and networking
    sudo apt install -y \
        libsqlite3-dev \
        libssl-dev \
        libffi-dev \
        libxml2-dev \
        libxslt1-dev
    
    # Graphics and GUI (headless)
    sudo apt install -y \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
        zlib1g-dev
    
    # SDL for pygame
    sudo apt install -y \
        libsdl2-dev \
        libsdl2-image-dev \
        libsdl2-mixer-dev \
        libsdl2-ttf-dev
    
    # Git libraries
    sudo apt install -y \
        libgit2-dev
    
    # Additional utilities
    sudo apt install -y \
        htop \
        tmux \
        screen \
        vim
    
    print_success "System dependencies installed"
}

# Create virtual environment
create_venv() {
    if [[ -d "$VENV_PATH" ]]; then
        print_warning "Virtual environment already exists at $VENV_PATH"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing virtual environment..."
            rm -rf "$VENV_PATH"
        else
            print_status "Using existing virtual environment"
            return 0
        fi
    fi
    
    print_status "Creating virtual environment with --system-site-packages..."
    python3 -m venv --system-site-packages "$VENV_PATH"
    
    # Update pip, setuptools, wheel
    print_status "Upgrading pip, setuptools, wheel..."
    "$VENV_PATH/bin/pip" install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created at $VENV_PATH"
}

# Install sentence-transformers with ARM64 fix
install_sentence_transformers() {
    print_status "Installing sentence-transformers (ARM64 optimized)..."
    
    source "$VENV_PATH/bin/activate"
    
    # Install dependencies first
    pip install numpy scipy scikit-learn
    
    # Install sentence-transformers without dependencies to avoid conflicts
    pip install --no-deps sentence-transformers
    
    # Install missing runtime dependencies manually
    pip install transformers tqdm torch huggingface-hub
    
    print_success "sentence-transformers installed with ARM64 compatibility"
}

# Install Python packages in dependency order
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Set pip configuration for ARM64
    export PIP_EXTRA_INDEX_URL="https://www.piwheels.org/simple"
    export PIP_FIND_LINKS="https://www.piwheels.org"
    
    # Install build dependencies first
    pip install --upgrade cython pybind11
    
    # Install PyTorch ecosystem (ARM64 specific)
    print_status "Installing PyTorch for ARM64..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # Install scientific computing packages
    print_status "Installing scientific packages..."
    pip install numpy scipy sympy mpmath
    
    # Install data processing and ML
    print_status "Installing data processing packages..."
    pip install pandas scikit-learn
    
    # Install sentence-transformers with special handling
    install_sentence_transformers
    
    # Install web and API
    print_status "Installing web packages..."
    pip install requests httpx aiofiles
    
    # Install database
    print_status "Installing database packages..."
    pip install chromadb sqlparse
    
    # Install visualization
    print_status "Installing visualization packages..."
    pip install matplotlib plotly
    
    # Install Streamlit and components
    print_status "Installing Streamlit and components..."
    pip install streamlit streamlit-monaco streamlit-ace
    
    # Install game and network libraries
    print_status "Installing specialized libraries..."
    pip install pygame chess networkx
    
    # Install utility packages
    print_status "Installing utility packages..."
    pip install jsbeautifier black python-dotenv passlib PyYAML marshmallow
    
    # Install linting and parsing
    print_status "Installing linting tools..."
    pip install sqlparse
    
    # Install security and crypto
    print_status "Installing security packages..."
    pip install RestrictedPython passlib
    
    # Install tokenization
    pip install tiktoken
    
    # Install remaining packages
    print_status "Installing remaining packages..."
    pip install \
        bs4 \
        nest-asyncio \
        ntplib \
        qutip \
        qiskit \
        pygit2 \
        openai \
        aiofiles
    
    # Install optional dependencies that may be missing
    pip install pillow sqlalchemy
    
    print_success "All Python packages installed"
}

# Create directory structure
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p "$SANDBOX_DIR"/{db,agents,config,viz}
    mkdir -p "$PROJECT_DIR"/prompts
    mkdir -p "$PROJECT_DIR"/logs
    
    # Create .env file template
    if [[ ! -f "$PROJECT_DIR/.env" ]]; then
        cat > "$PROJECT_DIR/.env" << EOF
# Moonshot API Configuration
MOONSHOT_API_KEY=your_api_key_here

# Application Settings
PROFILE_MODE=false
TEST_MODE=false

# Database Paths
DB_PATH=./sandbox/db/chatapp.db
CHROMA_PATH=./sandbox/db/chroma_db

# Logging
LOG_LEVEL=INFO
EOF
        print_success "Created .env template"
    fi
    
    # Create default prompts
    if [[ ! -f "$PROJECT_DIR/prompts/default.txt" ]]; then
        echo "You are Apex Aurum, a highly intelligent AI assistant powered by Moonshot AI." > "$PROJECT_DIR/prompts/default.txt"
    fi
    
    print_success "Directory structure created"
}

# Set permissions
set_permissions() {
    print_status "Setting permissions..."
    
    # Make the main script executable
    if [[ -f "$PROJECT_DIR/app.py" ]]; then
        chmod +x "$PROJECT_DIR/app.py"
    fi
    
    # Set ownership to current user
    sudo chown -R "$(whoami):$(whoami)" "$PROJECT_DIR"
    
    print_success "Permissions set"
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    source "$VENV_PATH/bin/activate"
    
    # Check critical packages
    local critical_packages=(streamlit torch numpy chromadb RestrictedPython)
    local failed_packages=()
    
    for package in "${critical_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            print_success "$package is installed"
        else
            print_error "$package failed to import"
            failed_packages+=("$package")
        fi
    done
    
    # Special handling for sentence-transformers
    print_status "Testing sentence-transformers import..."
    if python3 -c "import sentence_transformers" 2>/dev/null; then
        print_success "sentence-transformers is working"
    else
        print_warning "sentence-transformers import failed - trying manual install..."
        install_sentence_transformers
        # Test again
        if python3 -c "import sentence_transformers" 2>/dev/null; then
            print_success "sentence-transformers fixed successfully"
        else
            print_error "sentence-transformers still failing - manual intervention needed"
            failed_packages+=("sentence-transformers")
        fi
    fi
    
    if [[ ${#failed_packages[@]} -eq 0 ]]; then
        print_success "All critical packages verified"
    else
        print_warning "Some packages failed: ${failed_packages[*]}"
        print_status "You may need to install them manually or check for ARM compatibility"
    fi
}

# Print next steps
print_next_steps() {
    print_success "Installation complete!"
    echo
    echo -e "${BLUE}=== NEXT STEPS ===${NC}"
    echo "1. Activate the virtual environment:"
    echo "   source $VENV_PATH/bin/activate"
    echo
    echo "2. Edit the .env file with your Moonshot API key:"
    echo "   nano $PROJECT_DIR/.env"
    echo
    echo "3. Run the application:"
    echo "   streamlit run $PROJECT_DIR/app.py"
    echo
    echo -e "${YELLOW}NOTES:${NC}"
    echo "- cProfile and pstats are part of Python's standard library - removed from pip install"
    echo "- sentence-transformers has ARM64-specific handling to ensure compatibility"
    echo
    echo -e "${BLUE}TIPS FOR RASPBERRY PI:${NC}"
    echo "- Use 'htop' to monitor CPU/Memory usage"
    echo "- The first run may take longer as models are downloaded"
    echo "- Consider using a USB3 SSD for better performance with ChromaDB"
}

# Main installation flow
main() {
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  Apex Aurum - Raspberry Pi 5 Installation Script          ║${NC}"
    echo -e "${BLUE}║  FIXED: libatlas, Removed: cProfile/pstats, ARM64 fixes   ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    check_pi
    
    read -p "This will install system packages and create a virtual environment. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installation cancelled"
        exit 0
    fi
    
    update_system
    install_system_dependencies
    
    if ! check_python_version; then
        print_warning "Building Python $PYTHON_VERSION from source will take 20-30 minutes"
        read -p "Continue with source build? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_python_from_source
        else
            print_error "Python $PYTHON_VERSION is required but not installed"
            exit 1
        fi
    fi
    
    create_venv
    install_python_packages
    create_directories
    set_permissions
    verify_installation
    print_next_steps
}

# Handle script interruption
trap 'print_error "Installation interrupted"; exit 1' INT TERM

# Run main function
main "$@"
