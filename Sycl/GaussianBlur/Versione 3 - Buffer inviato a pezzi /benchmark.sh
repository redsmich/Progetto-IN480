#!/bin/bash

# ==============================================================================
#                               CONFIGURAZIONE
# ==============================================================================

# 1. Lista dei Kernel da testare (separati da spazio)
#KERNELS_TO_TEST=(301 501 701 901 1101 1301)
KERNELS_TO_TEST=(101 201)

# 2. Percorso dell'immagine da elaborare
IMG_PATH="images/pappagallone.jpg"

# 3. Dispositivo da usare: 0 = GPU, 1 = CPU
USE_CPU=0

# 4. Quante volte ripetere il test per OGNI kernel?
NUM_RUNS=5

# 5. Nome del file dove salvare i risultati
OUTPUT_FILE="dati_raw.csv"

# 6. Configurazione Ambiente
#export LD_LIBRARY_PATH=~/local/libs_fake:/opt/rocm-7.1.0/lib:~/local/llvm20-full/lib:$LD_LIBRARY_PATH

# ==============================================================================
#                               FINE CONFIGURAZIONE
# ==============================================================================

# Determiniamo il nome del dispositivo per il CSV
DEVICE_NAME="GPU"
if [ "$USE_CPU" -eq "1" ]; then
    DEVICE_NAME="CPU"
fi

# GESTIONE FILE CSV (APPEND vs OVERWRITE)
# Controlliamo se il file esiste già.
if [ ! -f "$OUTPUT_FILE" ]; then
    # Se NON esiste, lo creiamo e scriviamo l'intestazione
    echo "Kernel,Device,Image,Run_Number,Time_Seconds" > "$OUTPUT_FILE"
    echo "Creato nuovo file: $OUTPUT_FILE"
else
    # Se esiste già, non facciamo nulla (i dati verranno appesi sotto)
    echo "Il file $OUTPUT_FILE esiste già. I nuovi dati verranno aggiunti in coda."
fi

echo "=================================================="
echo "AVVIO BENCHMARK"
echo "Dispositivo: $DEVICE_NAME"
echo "Immagine: $IMG_PATH"
echo "=================================================="

# Ciclo sui Kernel
for KERNEL in "${KERNELS_TO_TEST[@]}"; do
    
    echo ""
    echo ">>> Testando Kernel Size: $KERNEL"

    # Ciclo sulle ripetizioni (Run)
    for ((i=1; i<=NUM_RUNS; i++)); do
        
        # Esecuzione del programma C++ con input iniettato
        output=$(echo -e "$KERNEL\n$IMG_PATH\n$USE_CPU\n0" | ./gaussian_blur)

        # Estrazione del tempo
        # 1. grep prende la riga
        # 2. awk prende il 5° elemento (assumendo sia il numero)
        # 3. sed toglie eventuali 's' finali se attaccate
        # 4. tr sostituisce il punto con la virgola
        tempo_raw=$(echo "$output" | grep "Tempo di puro calcolo" | awk '{print $5}')
        
        # Pulizia e conversione in virgola
        tempo_clean=$(echo "$tempo_raw" | sed 's/s//g' | tr '.' ',')

        # Controllo errori
        if [ -z "$tempo_clean" ]; then
            tempo_clean="ERROR"
            echo "   Run $i: ERRORE durante l'esecuzione"
        else
            echo "   Run $i: $tempo_clean s"
        fi

        # Salvataggio riga nel CSV (>> significa append, aggiungi in coda)
        echo "$KERNEL,$DEVICE_NAME,$IMG_PATH,$i,$tempo_clean" >> "$OUTPUT_FILE"
        
    done
done

echo ""
echo "=================================================="
echo "Finito! I dati aggiornati sono in $OUTPUT_FILE"
