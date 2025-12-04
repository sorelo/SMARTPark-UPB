import matplotlib.pyplot as plt
import os

# --- CONFIGURARE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_IMG = os.path.join(BASE_DIR, 'docs', 'datasets', 'dataset_distribution.png')

def count_images(folder):
    liber = len(os.listdir(os.path.join(folder, 'liber')))
    ocupat = len(os.listdir(os.path.join(folder, 'ocupat')))
    return liber, ocupat

def main():
    splits = ['train', 'validation', 'test']
    labels = []
    liber_counts = []
    ocupat_counts = []

    print("Generare statistici...")

    for split in splits:
        path = os.path.join(DATA_DIR, split)
        if not os.path.exists(path):
            print(f"Folder lipsă: {path}. Rulează split_dataset.py întâi!")
            return
        
        l, o = count_images(path)
        liber_counts.append(l)
        ocupat_counts.append(o)
        labels.append(split.capitalize())
        print(f"{split}: Liber={l}, Ocupat={o}")

    # Creare Grafic
    x = range(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([i - width/2 for i in x], liber_counts, width, label='Liber (0)', color='green')
    ax.bar([i + width/2 for i in x], ocupat_counts, width, label='Ocupat (1)', color='red')

    ax.set_ylabel('Număr Imagini')
    ax.set_title('Distribuția Dataset-ului SMARTPark')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG)
    print(f"Grafic salvat în: {OUTPUT_IMG}")
    # plt.show() # Decomentează dacă vrei să vezi graficul pe ecran

if __name__ == "__main__":
    main()