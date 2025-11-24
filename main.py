import sys

from src.train_model import train_and_save_model
from src.evaluate import evaluate_saved_model


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python main.py train     # Entrena y guarda modelo")
        print("  python main.py evaluate  # Evalúa modelo guardado")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        train_and_save_model()
    elif command == "evaluate":
        evaluate_saved_model()
    else:
        print(f"Comando no reconocido: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
