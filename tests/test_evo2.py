import torch
import platform
import sys

def evo2_embedding_test():
    from evo2 import Evo2
    device = 'cuda:0'
    evo2_model = Evo2('evo2_7b')

    sequence = 'ACGT'
    input_ids = torch.tensor(
        evo2_model.tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    layer_name = 'blocks.28.mlp.l3'

    outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

    print('Embeddings shape: ', embeddings[layer_name].shape)
    print('Outputs shape: ', outputs.shape)

if __name__ == "__main__":
    is_mac = platform.system() == 'Darwin'
    has_cuda = torch.cuda.is_available()
    if is_mac:
        print("[INFO] macOS detected — Evo2 requires transformer_engine which cannot be installed without CUDA.\n"
              "Your local version of Evo2 and the phllm repo automatically installs without CUDA, "
              "thus you cannot actually run Evo2 on your current machine.\n"
              "Please use a machine with access to CUDA environments.")
        print("Now exiting testing function.")
        sys.exit(0)

    if not has_cuda:
        print("[INFO] CUDA not available — Evo2's transformer_enginge requires CUDA, cannot proceed.")
        print("Now exiting testing function.")
        sys.exit(0)

    evo2_embedding_test()