{
    description = "Backdoor CNN Flake";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    };


    outputs = { self, nixpkgs }:
        let
            system = "x86_64-linux";
            pkgs = import nixpkgs { inherit system; };
        in {
            devShells.${system}.default = pkgs.mkShell {
                packages = with pkgs; [
                    python312
                    python312Packages.ruff
                    python312Packages.tkinter
                    rclone
                ];

                shellHook = ''
                    chmod +x ./scripts/sync-colab.sh
                    alias sync-colab="./scripts/sync-colab.sh"

                    chmod +x ./scripts/sync-weights.sh
                    alias sync-weights="./scripts/sync-weights.sh"

                    alias mk-pyenv="python -m venv .venv"
                    alias pyenv="source .venv/bin/activate"
                '';
            };
        };
}
