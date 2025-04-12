import time
from io import BytesIO
from pathlib import Path
import modal
import subprocess
import os


cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

cuda_dev_image = modal.Image.from_registry(
    f"nvidia/cuda:{tag}", add_python="3.10"
).entrypoint([])

lumina_image = (
    cuda_dev_image.apt_install(
        "git",
        "wget",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        "torchaudio==2.3.0",
        "pandas",
        "tensorboard",
        "fairscale",
        "sentencepiece",
        "gradio==4.19.0",
        "packaging",
        "transformers==4.46.2",
        "pyyaml",
        "pathlib",
        "Ninja",
        "bitsandbytes",
        "httpx[socks]",
        "einops",
        "regex",
        "h5py",
        "accelerate",
        "pre-commit",
        "torchao",
        "pytorch_lightning",
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
    )
    .run_commands(
        "git clone https://github.com/ayushnangia/Lumina-mGPT-2.0.git",
        "cd Lumina-mGPT-2.0 && pip install -e .",
        "mkdir -p /Lumina-mGPT-2.0/movqgan/270M", 
        "wget -O /Lumina-mGPT-2.0/movqgan/270M/movqgan_270M.ckpt https://huggingface.co/ai-forever/MoVQGAN/resolve/main/movqgan_270M.ckpt",
    )
)

app = modal.App("example-lumina", image=lumina_image)

MINUTES = 60  # seconds


@app.cls(
    gpu="H100",  # fastest GPU on Modal
    scaledown_window=10,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        # Example: Add a volume to persist generated samples
        "/persistent_samples": modal.Volume.from_name("lumina-samples", create_if_missing=True)
    },
)
class ImageGenerator:

    @modal.enter()
    def setup(self):
        """Optional: Code to run once when the container starts."""
        print("Container started.")
        # You could potentially load models here if needed,
        # but the generate.py script likely handles that.

    @modal.method()
    def generate(self):
        """Runs the Lumina generation script."""
        repo_path = "/Lumina-mGPT-2.0"
        script_path = f"{repo_path}/lumina_mgpt/generate_examples/generate.py"
        # Save path inside the container's ephemeral filesystem
        # ephemeral_save_path = "save_samples/"
        # Or, save to a persistent volume:
        persistent_save_path = "/persistent_samples/output"

    


        cmd = [
            "python", script_path,
            "--model_path", "Alpha-VLLM/Lumina-mGPT-2.0",
            "--save_path", persistent_save_path, # Use the relative temp path for the script
            # If saving to a persistent volume, you might need to adjust how save_path is handled
            # or copy files after generation: --save_path ephemeral_save_path
            "--cfg", "4.0",
            "--top_k", "4096",
            "--temperature", "1.0",
            "--width", "768",
            "--height", "768",
            "--speculative_jacobi",
            "--quant",
        ]

        print(f"Running command: {' '.join(cmd)}")
        print(f"Executing in: {repo_path}")
        print(f"Saving samples temporarily to: {full_script_save_path}")
        print(f"Final persistent save path: {persistent_save_path}")


        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=repo_path # Run script from the repository root
            )
            print("Generation successful.")
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)

            # Files are generated in full_script_save_path
            # Copy generated files from the temporary location to the persistent volume
            # generated_files_temp = os.listdir(full_script_save_path)
            # copied_files = []
            # for filename in generated_files_temp:
            #     source_path = os.path.join(full_script_save_path, filename)
            #     destination_path = os.path.join(persistent_save_path, filename)
            #     # Using shutil.move might be more efficient if files are large
            #     # For simplicity, using os.rename here
            #     os.rename(source_path, destination_path)
            #     copied_files.append(filename)
            #     print(f"Moved {filename} to {persistent_save_path}")


            # # Clean up temporary directory (optional)
            # # shutil.rmtree(full_script_save_path)

            # # List files from the persistent volume
            # final_generated_files = os.listdir(persistent_save_path)


            # print(f"Generated files moved to persistent storage: {copied_files}")
            # print(f"Files in persistent volume {persistent_save_path}: {final_generated_files}")


            # # Example: Return file contents (if small) or paths
            # # For now, just return status and file list
            # return {"status": "success", "files": copied_files, "persistent_save_path": persistent_save_path}

        except subprocess.CalledProcessError as e:
            print(f"Error during generation: {e}")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            return {"status": "error", "message": str(e), "stdout": e.stdout, "stderr": e.stderr}


    
