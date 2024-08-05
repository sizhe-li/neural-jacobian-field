from colorama import Fore

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"
