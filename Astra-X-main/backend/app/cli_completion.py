"""Shell completion for the ``astra`` CLI.

Provides ``astra completion`` to install shell completion for
bash, zsh, fish, and powershell.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["cmd_completion"]


BASH_COMPLETION = r'''# Astra X bash completion
_astra_completion() {
    local IFS=$'\n'
    local response
    response=$(ASTRA_COMPLETE=bash_comp "${COMP_WORDS[0]}" "${COMP_WORDS[@]:1}")
    for completion in $response; do
        COMPREPLY+=("$completion")
    done
}
complete -o default -F _astra_completion astra
'''

ZSH_COMPLETION = r'''# Astra X zsh completion
#compdef astra
_astra_completion() {
    local -a completions
    local -a completions_with_descriptions
    local -a response
    response=("${(@f)$(ASTRA_COMPLETE=zsh_comp "${words[@]}")}")
    for key in "${response[@]}"; do
        if [[ "$key" == --* ]]; then
            completions_with_descriptions+=("$key")
        else
            completions+=("$key")
        fi
    done
    if [ -n "$completions_with_descriptions" ]; then
        _describe -t completions_with_descriptions "completions" completions_with_descriptions
    fi
    if [ -n "$completions" ]; then
        compadd -a completions
    fi
}
compdef _astra_completion astra
'''

FISH_COMPLETION = r'''# Astra X fish completion
function _astra_completion
    set -l response (ASTRA_COMPLETE=fish_comp $argv[1] $argv[2-] | string split0)
    for completion in $response
        if string match -q '--*' $completion
            echo $completion
        else
            echo $completion
        end
    end
end
complete -c astra -f -a '(_astra_completion)'
'''

POWERSHELL_COMPLETION = r'''# Astra X PowerShell completion
Register-ArgumentCompleter -Native -CommandName 'astra' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    $env:ASTRA_COMPLETE = 'powershell_comp'
    $command = $commandAst.ToString()
    astra $command | ForEach-Object {
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }
    $env:ASTRA_COMPLETE = ''
}
'''

SHELL_COMPLETIONS = {
    "bash": BASH_COMPLETION,
    "zsh": ZSH_COMPLETION,
    "fish": FISH_COMPLETION,
    "powershell": POWERSHELL_COMPLETION,
}


def _detect_shell() -> str:
    """Detect the user's current shell."""
    import os
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    if "fish" in shell:
        return "fish"
    if "powershell" in shell or "pwsh" in shell:
        return "powershell"
    return "bash"


def cmd_completion(shell: str | None = None, install: bool = False) -> int:
    """Generate or install shell completion."""
    shell = shell or _detect_shell()
    shell = shell.lower()

    if shell not in SHELL_COMPLETIONS:
        print(f"Unsupported shell: {shell}. Supported: {', '.join(SHELL_COMPLETIONS.keys())}")
        return 1

    completion_script = SHELL_COMPLETIONS[shell]

    if not install:
        print(completion_script)
        return 0

    # Install completion
    home = Path.home()

    if shell == "bash":
        bashrc = home / ".bashrc"
        marker = "# Astra X completion"
        if marker not in bashrc.read_text():
            with open(bashrc, "a") as f:
                f.write(f"\n{marker}\n")
                f.write("eval \"$(astra completion bash)\"\n")
            print(f"Installed completion to {bashrc}. Restart shell or run: source {bashrc}")
        else:
            print("Completion already installed.")
    elif shell == "zsh":
        zshrc = home / ".zshrc"
        marker = "# Astra X completion"
        if marker not in zshrc.read_text():
            with open(zshrc, "a") as f:
                f.write(f"\n{marker}\n")
                f.write("eval \"$(astra completion zsh)\"\n")
            print(f"Installed completion to {zshrc}. Restart shell or run: source {zshrc}")
        else:
            print("Completion already installed.")
    elif shell == "fish":
        fish_dir = home / ".config" / "fish" / "completions"
        fish_dir.mkdir(parents=True, exist_ok=True)
        completion_file = fish_dir / "astra.fish"
        completion_file.write_text(completion_script)
        print(f"Installed completion to {completion_file}.")
    elif shell == "powershell":
        profile = home / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1"
        profile.parent.mkdir(parents=True, exist_ok=True)
        marker = "# Astra X completion"
        content = profile.read_text() if profile.exists() else ""
        if marker not in content:
            with open(profile, "a") as f:
                f.write(f"\n{marker}\n")
                f.write("Register-ArgumentCompleter -Native -CommandName 'astra' -ScriptBlock {\n")
                f.write("    param($wordToComplete, $commandAst, $cursorPosition)\n")
                f.write("    $env:ASTRA_COMPLETE = 'powershell_comp'\n")
                f.write("    $command = $commandAst.ToString()\n")
                f.write("    astra $command | ForEach-Object {\n")
                f.write("        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)\n")
                f.write("    }\n")
                f.write("    $env:ASTRA_COMPLETE = ''\n")
                f.write("}\n")
            print(f"Installed completion to {profile}. Restart PowerShell.")
        else:
            print("Completion already installed.")

    return 0
