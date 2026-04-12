#!/bin/bash
# Optimizer script: Find best kernel variant per matrix size

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_DIR="$PROJECT_DIR/results"

SIZES="64 128 256 512 1024 2048 4096"

declare -a VARIANTS=(
    "3  r1y"
    "4  r2x"
    "8  r2z2"
    "9  r3x"
)

echo "=== GEMM Kernel Optimizer ==="
echo "Testing: r1y r2x r2z2 r3x"
echo "Sizes: $SIZES"
echo ""

mkdir -p "$RESULTS_DIR"

declare -A custom_times
declare -A sgemm_times

for variant_info in "${VARIANTS[@]}"; do
    read variant name <<< "$variant_info"
    echo -n "Testing $name... "
    
    sed -i "s/#define FP32_VARIANT [0-9]*/#define FP32_VARIANT $variant/" "$PROJECT_DIR/src/main.cu"
    cd "$BUILD_DIR" && make -j$(nproc) > /dev/null 2>&1
    
    output=$("$BUILD_DIR/cu_x_gemm" 2>&1)
    
    # Extract FP32 section data lines (lines starting with a number, after "=== FP32 ===")
    fp32_section=$(echo "$output" | sed -n '/=== FP32 ===/,/^$/p' | awk '/^[[:space:]]*[0-9]/')
    
    for size in $SIZES; do
        line=$(echo "$fp32_section" | awk -v s="$size" '$1 == s')
        if [ -n "$line" ]; then
            # Fields: $1=Dim $2=Ver $3=Desc $4=Custom(ms) $5=Sgemm(ms) $6=CUDA(ms) ...
            custom=$(echo "$line" | awk '{print $4}')
            sgemm=$(echo "$line" | awk '{print $5}')
            custom_times[${name}_${size}]=$custom
            sgemm_times[${name}_${size}]=$sgemm
        fi
    done
    
    echo "done"
done

# Reset to default
sed -i "s/#define FP32_VARIANT [0-9]*/#define FP32_VARIANT 6/" "$PROJECT_DIR/src/main.cu"

# Output table
echo ""
echo "=== Results (ratio = sgemm/custom * 100%, >100% = we beat cuBLAS) ==="
echo ""
printf "%-8s %-8s %-8s %-8s %-8s %-8s %-10s\n" "Size" "r1y" "r2x" "r2z2" "r3x" "Best%" "Best"
echo "----------------------------------------------------------------------"

for size in $SIZES; do
    best_kernel=""
    best_ratio=0
    
    for variant_info in "${VARIANTS[@]}"; do
        read variant name <<< "$variant_info"
        
        custom=${custom_times[${name}_${size}]}
        sgemm=${sgemm_times[${name}_${size}]}
        
        if [ -n "$custom" ] && [ -n "$sgemm" ] && [ "$custom" != "0" ]; then
            ratio=$(awk "BEGIN {printf \"%.1f\", $sgemm/$custom*100}")
        else
            ratio="N/A"
        fi
        
        eval "ratio_$name='$ratio'"
        
        if [[ "$ratio" != "N/A" ]]; then
            is_better=$(awk -v R="$ratio" -v B="$best_ratio" 'BEGIN {print (R > B) ? 1 : 0}')
            if [ "$is_better" = "1" ]; then
                best_ratio=$ratio
                best_kernel=$name
            fi
        fi
    done
    
    printf "%-8s %-8s %-8s %-8s %-8s %-8s %-10s\n" \
        "$size" "${ratio_r1y:-N/A}" "${ratio_r2x:-N/A}" "${ratio_r2z2:-N/A}" "${ratio_r3x:-N/A}" "${best_ratio}%" "$best_kernel"
done

echo ""
echo "=== Recommended (Best performer for each size) ==="
for size in $SIZES; do
    best_kernel=""
    best_ratio=0
    
    for variant_info in "${VARIANTS[@]}"; do
        read variant name <<< "$variant_info"
        
        custom=${custom_times[${name}_${size}]}
        sgemm=${sgemm_times[${name}_${size}]}
        
        if [ -n "$custom" ] && [ -n "$sgemm" ] && [ "$custom" != "0" ]; then
            ratio=$(awk "BEGIN {printf \"%.1f\", $sgemm/$custom*100}")
            is_better=$(awk -v R="$ratio" -v B="$best_ratio" 'BEGIN {print (R > B) ? 1 : 0}')
            if [ "$is_better" = "1" ]; then
                best_ratio=$ratio
                best_kernel=$name
            fi
        fi
    done
    
    if [ -n "$best_kernel" ]; then
        beat_msg=""
        if [ "$(awk -v R="$best_ratio" 'BEGIN {print (R > 100) ? 1 : 0}')" = "1" ]; then
            beat_msg=" [BEATS CUBLAS]"
        fi
        echo "Size $size: best=$best_kernel (${best_ratio}%)$beat_msg"
    fi
done