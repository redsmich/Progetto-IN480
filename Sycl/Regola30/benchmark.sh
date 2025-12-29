#!/bin/bash

# ==============================================================================
#                               CONFIGURAZIONE
# ==============================================================================

# 1. Lista delle Larghezze (Width) da testare
#    Nota: L'altezza sarà calcolata automaticamente come Width / 2
WIDTHS_TO_TEST=(1024 2048 4096 8192 16384 32768 65536)

# 2. Dispositivo da usare: 0 = GPU, 1 = CPU
USE_CPU=0

# 3. Quante volte ripetere il test per OGNI dimensione?
NUM_RUNS=5

# 4. Nome del file dove salvare i risultati
OUTPUT_FILE="dati_automa.csv"

# 5. Nome dell'eseguibile (assicurati di averlo compilato!)
EXECUTABLE="./regola30"

# 6. Configurazione Ambiente (se serve, scommenta)
# source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

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
    echo "Il file $OUTPUT_FILE esiste già. I nuovi dati verranno aggiunti in coda."
fi

echo "=================================================="
echo "AVVIO BENCHMARK AUTOMA"
echo "Dispositivo: $DEVICE_NAME"
echo "Output: $OUTPUT_FILE"
echo "=================================================="

# Ciclo sulle Dimensioni (Width)
for WIDTH in "${WIDTHS_TO_TEST[@]}"; do
    
    # Calcoliamo l'altezza (Height) come metà della larghezza
    # (Puoi cambiare questa logica se vuoi, es. HEIGHT=$WIDTH)
    HEIGHT=$((WIDTH / 2))

    echo ""
    echo ">>> Testando Grid: $WIDTH x $HEIGHT"

    # Ciclo sulle ripetizioni (Run)
    for ((i=1; i<=NUM_RUNS; i++)); do
        
        # COSTRUZIONE INPUT PER IL C++
        # Ordine richiesto dal tuo codice:
        # 1. Width (cin >> width)
        # 2. Height (cin >> height)
        # 3. UseCpu (cin >> useCpu)
        # 4. PrintInfo (cin >> print) - Mettiamo 0
        
        INPUT_STRING="$WIDTH\n$HEIGHT\n$USE_CPU\n0"

        # Esecuzione
        output=$(echo -e "$INPUT_STRING" | $EXECUTABLE)

        # Estrazione del tempo
        # Cerca la riga "Tempo di puro calcolo: X secondi."
        tempo_raw=$(echo "$output" | grep "Tempo di puro calcolo" | awk '{print $5}')
        
        # Pulizia (via la 's' se c'è, punto in virgola)
        tempo_clean=$(echo "$tempo_raw" | sed 's/s//g' | tr '.' ',')

        # Controllo errori
        if [ -z "$tempo_clean" ]; then
            tempo_clean="ERROR"
            echo "   Run $i: ERRORE (Il programma potrebbe essere crashato o output non trovato)"
            # Debug: scommenta la riga sotto se vuoi vedere l'errore a video
            # echo "$output"
        else
            echo "   Run $i: $tempo_clean s"
        fi

        # Salvataggio nel CSV
        echo "$WIDTH,$HEIGHT,$DEVICE_NAME,$i,$tempo_clean" >> "$OUTPUT_FILE"
        
    done
done

echo ""
echo "=================================================="
echo "Finito! I dati sono in $OUTPUT_FILE"
