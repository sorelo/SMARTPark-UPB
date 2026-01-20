import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_IMG = os.path.join(BASE_DIR, 'docs', 'datasets', 'dataset_distribution.png')

def main():
    splits = ['train', 'validation', 'test']
    labels = ['liber', 'ocupat']
    
    data = {s: [len(os.listdir(os.path.join(DATA_DIR, s, l))) for l in labels] for s in splits}

    x = range(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    liber_vals = [data[s][0] for s in splits]
    ocupat_vals = [data[s][1] for s in splits]

    ax.bar([i - width/2 for i in x], liber_vals, width, label='Liber (0)', color='#DEFF9A', edgecolor='black')
    ax.bar([i + width/2 for i in x], ocupat_vals, width, label='Ocupat (1)', color='#FF6B6B', edgecolor='black')

    ax.set_ylabel('Număr de imagini (Samples)')
    ax.set_title('Distribuția Dataset-ului SMARTPark UPB (Date Sintetice)')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits])
    ax.legend()

    # Adăugăm cifrele deasupra barelor
    for i, val in enumerate(liber_vals): ax.text(i - width/2, val + 5, str(val), ha='center')
    for i, val in enumerate(ocupat_vals): ax.text(i + width/2, val + 5, str(val), ha='center')

    os.makedirs(os.path.dirname(OUTPUT_IMG), exist_ok=True)
    plt.savefig(OUTPUT_IMG)
    print(f"Graficul a fost salvat în: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()