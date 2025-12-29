#!/bin/bash

# ==============================================================================
#                               CONFIGURAZIONE
# ==============================================================================

# 1. Lista dei Kernel da testare
#KERNELS_TO_TEST=(301 501 701 901 1101 1301)
KERNELS_TO_TEST=(101 201)
# 2. Percorso dell'immagine
IMG_PATH="images/pappagallone.jpg"

# 3. Dispositivo: 0 = GPU, 1 = CPU (Nel tuo codice Vulkan 0 cerca GPU)
USE_CPU=0

# 4. Ripetizioni
NUM_RUNS=5

# 5. Nome file output
OUTPUT_FILE="dati_vulkan.csv"

# 6. Nome eseguibile Vulkan
EXECUTABLE="./gaussian_blur"

# ==============================================================================
#                               FINE CONFIGURAZIONE
# ==============================================================================

# Determiniamo il nome del dispositivo per il CSV
DEVICE_NAME="GPU"
if [ "$USE_CPU" -eq "1" ]; then
    DEVICE_NAME="CPU"
fi

# GESTIONE FILE CSV
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "Kernel,Device,Image,Run_Number,Time_Seconds" > "$OUTPUT_FILE"
    echo "Creato nuovo file: $OUTPUT_FILE"
else
    echo "Il file $OUTPUT_FILE esiste già. Append..."
fi

echo "=================================================="
echo "AVVIO BENCHMARK VULKAN"
echo "Dispositivo: $DEVICE_NAME"
echo "Immagine: $IMG_PATH"
echo "=================================================="

# Ciclo sui Kernel
for KERNEL in "${KERNELS_TO_TEST[@]}"; do
    
    echo ""
    echo ">>> Testando Kernel: $KERNEL"

    for ((i=1; i<=NUM_RUNS; i++)); do
        
        # Ordine Input Vulkan C++:
        # 1. Kernel Size (cin >> kerDim)
        # 2. Path (cin >> pathName)
        # 3. CPU/GPU (cin >> useCpu)
        # 4. Print Devices (cin >> print) -> Mettiamo 0
        
        INPUT_STRING="$KERNEL\n$IMG_PATH\n$USE_CPU\n0"

        output=$(echo -e "$INPUT_STRING" | $EXECUTABLE)

        # Parsing Output
        # Cerca: "Tempo di puro calcolo: X secondi"
        tempo_raw=$(echo "$output" | grep "Tempo di puro calcolo" | awk '{print $5}')
        
        # Pulizia
        tempo_clean=$(echo "$tempo_raw" | sed 's/s//g' | tr '.' ',')

        if [ -z "$tempo_clean" ]; then
            tempo_clean="ERROR"
            echo "   Run $i: ERRORE (Output non trovato o crash)"
            # echo "$output" # Decommenta per debug
        else
            echo "   Run $i: $tempo_clean s"
        fi

        echo "$KERNEL,$DEVICE_NAME,$IMG_PATH,$i,$tempo_clean" >> "$OUTPUT_FILE"
        
    done
done

echo ""
echo "=================================================="
echo "Finito! Dati salvati in $OUTPUT_FILE"
