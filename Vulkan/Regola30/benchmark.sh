#!/bin/bash

# ==============================================================================
#                               CONFIGURAZIONE
# ==============================================================================

# 1. Lista delle Larghezze (Width) da testare
#    L'altezza (Iterazioni) sarà calcolata automaticamente come Width / 2
WIDTHS_TO_TEST=(1024 2048 4096 8192 16384 32768 65536)
#WIDTHS_TO_TEST=(65536)

# 2. Dispositivo: 0 = GPU, 1 = CPU
USE_CPU=0

# 3. Ripetizioni per ogni dimensione
NUM_RUNS=5

# 4. Nome file output
OUTPUT_FILE="dati_regola30.csv"

# 5. Nome eseguibile Vulkan
EXECUTABLE="./regola30"

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
    echo "Width,Height,Device,Run_Number,Time_Seconds" > "$OUTPUT_FILE"
    echo "Creato nuovo file: $OUTPUT_FILE"
else
    echo "Il file $OUTPUT_FILE esiste già. Append..."
fi

echo "=================================================="
echo "AVVIO BENCHMARK REGOLA 30 (VULKAN)"
echo "Dispositivo: $DEVICE_NAME"
echo "=================================================="

# Ciclo sulle Larghezze
for WIDTH in "${WIDTHS_TO_TEST[@]}"; do
    
    # Calcolo Altezza (Iterazioni) = Larghezza / 2
    HEIGHT=$((WIDTH / 2))

    echo ""
    echo ">>> Testando Grid: $WIDTH x $HEIGHT"

    for ((i=1; i<=NUM_RUNS; i++)); do
        
        # Ordine Input nel tuo C++ (funzione input e choose_and_printDevices):
        # 1. Width (cin >> tmp1)
        # 2. Height/Iterazioni (cin >> tmp2)
        # 3. CPU/GPU (cin >> useCpu)
        # 4. Print Devices (cin >> print) -> Mettiamo 0
        
        INPUT_STRING="$WIDTH\n$HEIGHT\n$USE_CPU\n0"

        output=$(echo -e "$INPUT_STRING" | $EXECUTABLE)

        # Parsing Output
        # Cerca: "Tempo di puro calcolo: X secondi"
        tempo_raw=$(echo "$output" | grep "Tempo di puro calcolo" | awk '{print $5}')
        
        # Pulizia (rimuove 's' se presente e cambia punto in virgola)
        tempo_clean=$(echo "$tempo_raw" | sed 's/s//g' | tr '.' ',')

        if [ -z "$tempo_clean" ]; then
            tempo_clean="ERROR"
            echo "   Run $i: ERRORE (Output non trovato o crash)"
            # echo "$output" # Decommenta per debug in caso di crash
        else
            echo "   Run $i: $tempo_clean s"
        fi

        # Salvataggio CSV
        echo "$WIDTH,$HEIGHT,$DEVICE_NAME,$i,$tempo_clean" >> "$OUTPUT_FILE"
        
    done
done

echo ""
echo "=================================================="
echo "Finito! Dati salvati in $OUTPUT_FILE"
